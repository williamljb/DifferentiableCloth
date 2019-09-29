/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "collision.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "optimization.hpp"

#include "simulation.hpp"
#include "timer.hpp"
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <vector>
#include "alglib/linalg.h"
#include "alglib/solvers.h"
using namespace std;
using namespace alglib;
#include <torch/torch.h>
using namespace torch;

#include <algorithm>
#include <utility>

static const int max_iter = 100;
static const Tensor &thickness = ::magic.projection_thickness;

static Tensor obs_mass;
static bool deform_obstacles;

static vector<Tensor> xold;
static vector<Tensor> xold_obs;

namespace CO {

Tensor get_mass (const Node *node) {return is_free(node) ? node->m : obs_mass;}

// returns pair of (i) is_free(vert), and
// (ii) index of mesh in ::meshes or ::obs_meshes that contains vert
pair<bool,int> find_in_meshes (const Node *node) {
    int m = find_mesh(node, *::meshes);
    if (m != -1)
        return make_pair(true, m);
    else
        return make_pair(false, find_mesh(node, *::obs_meshes));
}

void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones);

vector<Impact> find_impacts (const vector<AccelStruct*> &acc,
                             const vector<AccelStruct*> &obs_accs);
vector<Impact> independent_impacts (const vector<Impact> &impacts);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones);

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons, bool verbose = false);

vector<Constraint> impact_constraints (const vector<ImpactZone*> &zones);

ostream &operator<< (ostream &out, const Impact &imp);
ostream &operator<< (ostream &out, const ImpactZone *zone);

void collision_response (vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    ::meshes = &meshes;
    ::obs_meshes = &obs_meshes;
    ::xold = node_positions(meshes);
    ::xold_obs = node_positions(obs_meshes);
    // cout << "create bvh" << endl;
    vector<AccelStruct*> accs = create_accel_structs(meshes, true),
                         obs_accs = create_accel_structs(obs_meshes, true);
    vector<ImpactZone*> zones, prezones;
    ::obs_mass = ONE*1e3;
    int iter;
    bool changed = false;
    for (int deform = 0; deform <= 1; deform++) {
        zones.clear();prezones.clear();
        ::deform_obstacles = deform;
        for (iter = 0; iter < max_iter; iter++) {
            zones.clear();
            for (auto p : prezones) {
                ImpactZone *newp = new ImpactZone;
                *newp = *p;
                zones.push_back(newp);
            }
            for (auto p : prezones)
                if (!p->active) //true for non-backward purposes
                    delete p;
            if (!zones.empty())
                update_active(accs, obs_accs, zones);
    // cout << "find_impacts" << endl;
            vector<Impact> impacts = find_impacts(accs, obs_accs);
            impacts = independent_impacts(impacts);
            if (impacts.empty())
                break;
            add_impacts(impacts, zones);
    // cout << "apply_inelastic_projection" << endl;
            for (int z = 0; z < zones.size(); z++) {
                ImpactZone *zone = zones[z];
                if (zone->active){
                    changed = true;
                    apply_inelastic_projection(zone, cons, verbose);
                }
            }
    // cout << "update_accel_struct" << endl;
            for (int a = 0; a < accs.size(); a++)
                update_accel_struct(*accs[a]);
            for (int a = 0; a < obs_accs.size(); a++)
                update_accel_struct(*obs_accs[a]);
            if (deform_obstacles)
                ::obs_mass /= 2;
            prezones = zones;
        }
        if (iter < max_iter) // success!
            break;
    }
    if (iter == max_iter) {
        cerr << "Collision resolution failed to converge!" << endl;
        debug_save_meshes(meshes, "meshes");
        debug_save_meshes(obs_meshes, "obsmeshes");
        exit(1);
    }
        for (int m = 0; m < meshes.size(); m++) {
if (changed)
            compute_ws_data(*meshes[m]);
            update_x0(*meshes[m]);
        }
        for (int o = 0; o < obs_meshes.size(); o++) {
if (changed)
            compute_ws_data(*obs_meshes[o]);
            update_x0(*obs_meshes[o]);
        }
    for (int z = 0; z < zones.size(); z++)
        delete zones[z];
    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);
    // cout << "collision end" << endl;
}

void update_active (const vector<AccelStruct*> &accs,
                    const vector<AccelStruct*> &obs_accs,
                    const vector<ImpactZone*> &zones) {
    for (int a = 0; a < accs.size(); a++)
        mark_all_inactive(*accs[a]);
    for (int a = 0; a < obs_accs.size(); a++)
        mark_all_inactive(*obs_accs[a]);
    for (int z = 0; z < zones.size(); z++) {
        const ImpactZone *zone = zones[z];
        if (!zone->active)
            continue;
        for (int n = 0; n < zone->nodes.size(); n++) {
            const Node *node = zone->nodes[n];
            pair<bool,int> mi = find_in_meshes(node);
            AccelStruct *acc = (mi.first ? accs : obs_accs)[mi.second];
            for (int v = 0; v < node->verts.size(); v++)
                for (int f = 0; f < node->verts[v]->adjf.size(); f++)
                    mark_active(*acc, node->verts[v]->adjf[f]);
        }
    }
}

// Impacts

static int nthreads = 0;
static vector<Impact> *impacts = NULL;
static vector<pair<Face const*, Face const*> > *faceimpacts = NULL;
static int *cnt = NULL;

void find_face_impacts (const Face *face0, const Face *face1);

bool vf_collision_test (const Vert *vert, const Face *face, Impact &impact);
bool ee_collision_test (const Edge *edge0, const Edge *edge1, Impact &impact);
bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact);

void compute_face_impacts (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    Impact impact;
    BOX nb[6], eb[6], fb[2];
    for (int v = 0; v < 3; ++v) {
        nb[v] = node_box(face0->v[v]->node, true);
        nb[v+3] = node_box(face1->v[v]->node, true);
    }
    for (int v = 0; v < 3; ++v) {
        eb[v] = nb[NEXT(v)]+nb[PREV(v)];//edge_box(face0->adje[v], true);//
        eb[v+3] = nb[NEXT(v)+3]+nb[PREV(v)+3];//edge_box(face1->adje[v], true);//
    }
    fb[0] = nb[0]+nb[1]+nb[2];
    fb[1] = nb[3]+nb[4]+nb[5];
    double thick = ::thickness.item<double>();
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v], fb[1], thick))
            continue;
        if (vf_collision_test(face0->v[v], face1, impact))
            CO::impacts[t].push_back(impact);
    }
    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v+3], fb[0], thick))
            continue;
        if (vf_collision_test(face1->v[v], face0, impact))
            CO::impacts[t].push_back(impact);
    }
    for (int e0 = 0; e0 < 3; e0++)
        for (int e1 = 0; e1 < 3; e1++) {
            if (!overlap(eb[e0], eb[e1+3], thick))
                continue;
            if (ee_collision_test(face0->adje[e0], face1->adje[e1], impact))
                CO::impacts[t].push_back(impact);
        }
}

vector<Impact> find_impacts (const vector<AccelStruct*> &accs,
                             const vector<AccelStruct*> &obs_accs) {
    if (!impacts) {
        CO::nthreads = omp_get_max_threads();
        CO::impacts = new vector<Impact>[CO::nthreads];
        CO::faceimpacts = new vector<pair<Face const*, Face const*> >[CO::nthreads];
        CO::cnt = new int[CO::nthreads];
    }
    for (int t = 0; t < CO::nthreads; t++) {
        CO::impacts[t].clear();
        CO::faceimpacts[t].clear();
        CO::cnt[t] = 0;
    }
    for_overlapping_faces(accs, obs_accs, ::thickness, find_face_impacts);
    vector<pair<Face const*, Face const*> > tot_faces;
    for (int t = 0; t < CO::nthreads; ++t)
        append(tot_faces, CO::faceimpacts[t]);
    random_shuffle(tot_faces.begin(), tot_faces.end());
    // #pragma omp parallel for
    for (int i = 0; i < tot_faces.size(); ++i)
        compute_face_impacts(tot_faces[i].first,tot_faces[i].second);
    vector<Impact> impacts;
    for (int t = 0; t < CO::nthreads; t++) {
        // cout << CO::cnt[t] << "->" << CO::impacts[t].size() << endl;
        append(impacts, CO::impacts[t]);
    }
    return impacts;
}

void find_face_impacts (const Face *face0, const Face *face1) {
    int t = omp_get_thread_num();
    CO::faceimpacts[t].push_back(make_pair(face0, face1));
}

bool vf_collision_test (const Vert *vert, const Face *face, Impact &impact) {
    const Node *node = vert->node;
    if (node == face->v[0]->node
     || node == face->v[1]->node
     || node == face->v[2]->node)
        return false;
    return collision_test(Impact::VF, node, face->v[0]->node, face->v[1]->node,
                          face->v[2]->node, impact);
}

bool ee_collision_test (const Edge *edge0, const Edge *edge1, Impact &impact) {
    if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
        || edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
        return false;
    return collision_test(Impact::EE, edge0->n[0], edge0->n[1],
                          edge1->n[0], edge1->n[1], impact);
}

Tensor pos (const Node *node, Tensor t) {return node->x0 + t*(node->x - node->x0);}

bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact) {
    int t0 = omp_get_thread_num();
        ++CO::cnt[t0];
    impact.type = type;
    impact.nodes[0] = (Node*)node0;
    impact.nodes[1] = (Node*)node1;
    impact.nodes[2] = (Node*)node2;
    impact.nodes[3] = (Node*)node3;
    const Tensor &x0 = node0->x0, v0 = node0->x - x0;
    Tensor x1 = node1->x0 - x0, x2 = node2->x0 - x0, x3 = node3->x0 - x0;
    Tensor v1 = (node1->x - node1->x0) - v0, v2 = (node2->x - node2->x0) - v0,
         v3 = (node3->x - node3->x0) - v0;
    Tensor a0 = stp(x1, x2, x3),
           a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
           a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
           a3 = stp(v1, v2, v3);
    Tensor t = solve_cubic(a3, a2, a1, a0);
    // t = cat({t, ONE.reshape({1})}, 0);
    int nsol = t.size(0);
    // t[nsol] = 1; // also check at end of timestep
    for (int i = 0; i < nsol; i++) {
        if ((t[i] < 0).item<int>() || (t[i] > 1).item<int>())
            continue;
        impact.t = t[i];
        Tensor bx0 = x0+t[i]*v0, bx1 = x1+t[i]*v1,
             bx2 = x2+t[i]*v2, bx3 = x3+t[i]*v3;
        Tensor &n = impact.n;
        Tensor *w = impact.w;
        w[0] = w[1] = w[2] = w[3] = ZERO;
        Tensor d;
        bool inside, over = false;
        if (type == Impact::VF) {
            d = sub_signed_vf_distance(bx1, bx2, bx3, &n, w, 1e-6, over);
            inside = (torch::min(-w[1], torch::min(-w[2], -w[3])) >= -1e-6).item<int>();
        } else {// Impact::EE
            d = sub_signed_ee_distance(bx1, bx2, bx3, bx2-bx1, bx3-bx1, bx3-bx2, &n, w, 1e-6, over);
            inside = (torch::min(torch::min(w[0], w[1]), torch::min(-w[2], -w[3])) >= -1e-6).item<int>();
        }
        if (over || !inside)
            continue;
        if ((dot(n, w[1]*v1 + w[2]*v2 + w[3]*v3) > 0).item<int>())
            n = -n;
        // if ((abs(d)<1e-6).item<int>() && inside)
        //     return true;
        return true;
    }
    return false;
}

// Independent impacts

bool operator< (const Impact &impact0, const Impact &impact1) {
    return (impact0.t < impact1.t).item<int>();
}

bool conflict (const Impact &impact0, const Impact &impact1);

vector<Impact> independent_impacts (const vector<Impact> &impacts) {
    vector<Impact> sorted = impacts;
    sort(sorted.begin(), sorted.end());
    vector<Impact> indep;
    for (int e = 0; e < sorted.size(); e++) {
        const Impact &impact = sorted[e];
        bool con = false;
        for (int e1 = 0; e1 < indep.size(); e1++)
            if (conflict(impact, indep[e1]))
                con = true;
        if (!con)
            indep.push_back(impact);
    }
    return indep;
}

bool conflict (const Impact &i0, const Impact &i1) {
    return (is_free(i0.nodes[0]) && is_in(i0.nodes[0], i1.nodes, 4))
        || (is_free(i0.nodes[1]) && is_in(i0.nodes[1], i1.nodes, 4))
        || (is_free(i0.nodes[2]) && is_in(i0.nodes[2], i1.nodes, 4))
        || (is_free(i0.nodes[3]) && is_in(i0.nodes[3], i1.nodes, 4));
}

// Impact zones

ImpactZone *find_or_create_zone (const Node *node, vector<ImpactZone*> &zones);
void merge_zones (ImpactZone* zone0, ImpactZone *zone1,
                  vector<ImpactZone*> &zones);

void add_impacts (const vector<Impact> &impacts, vector<ImpactZone*> &zones) {
    for (int z = 0; z < zones.size(); z++)
        zones[z]->active = false;
    for (int i = 0; i < impacts.size(); i++) {
        const Impact &impact = impacts[i];
        Node *node = impact.nodes[is_free(impact.nodes[0]) ? 0 : 3];
        ImpactZone *zone = find_or_create_zone(node, zones);
        for (int n = 0; n < 4; n++)
            if (is_free(impact.nodes[n]) || ::deform_obstacles)
                merge_zones(zone, find_or_create_zone(impact.nodes[n], zones),
                            zones);
        zone->impacts.push_back(impact);
        zone->active = true;
    }
}

ImpactZone *find_or_create_zone (const Node *node, vector<ImpactZone*> &zones) {
    for (int z = 0; z < zones.size(); z++)
        if (is_in((Node*)node, zones[z]->nodes))
            return zones[z];
    ImpactZone *zone = new ImpactZone;
    zone->nodes.push_back((Node*)node);
    zones.push_back(zone);
    return zone;
}

void merge_zones (ImpactZone* zone0, ImpactZone *zone1,
                  vector<ImpactZone*> &zones) {
    if (zone0 == zone1)
        return;
    append(zone0->nodes, zone1->nodes);
    append(zone0->impacts, zone1->impacts);
    exclude(zone1, zones);
    delete zone1;
}

// Response

struct NormalOpt: public NLConOpt {
    ImpactZone *zone;
    Tensor inv_m;
    vector<double> tmp;
    NormalOpt (): zone(NULL), inv_m(ZERO) {nvar = ncon = 0;}
    NormalOpt (ImpactZone *zone): zone(zone), inv_m(ZERO) {
        nvar = zone->nodes.size()*3;
        ncon = zone->impacts.size();
        for (int n = 0; n < zone->nodes.size(); n++)
            inv_m = inv_m + 1/get_mass(zone->nodes[n]);
        inv_m = inv_m / (double)zone->nodes.size();
        tmp = vector<double>(nvar);
    }
    void initialize (double *x) const;
    void precompute (const double *x) const;
    double objective (const double *x) const;
    void obj_grad (const double *x, double *grad) const;
    double constraint (const double *x, int i, int &sign) const;
    void con_grad (const double *x, int i, double factor, double *grad) const;
    void finalize (const double *x);
};

Tensor &get_xold (const Node *node);

void precompute_derivative(real_2d_array &a, real_2d_array &q, real_2d_array &r0, vector<double> &lambda,
                            real_1d_array &sm_1, vector<int> &legals, double **grads, ImpactZone *zone,
                            NormalOpt &slx) {
    a.setlength(slx.nvar,legals.size());
    sm_1.setlength(slx.nvar);
    for (int i = 0; i < slx.nvar; ++i)
        sm_1[i] = 1.0/sqrt(slx.inv_m*get_mass(zone->nodes[i/3])).item<double>();
    for (int k = 0; k < legals.size(); ++k)
        for (int i = 0; i < slx.nvar; ++i)
            a[i][k]=grads[legals[k]][i] * sm_1[i]; //sqrt(m^-1)
    real_1d_array tau, r1lam1, lamp;
    tau.setlength(slx.nvar);
    rmatrixqr(a, slx.nvar, legals.size(), tau);
    real_2d_array qtmp, r, r1;
int cols = legals.size();
if (cols>slx.nvar)cols=slx.nvar;
    rmatrixqrunpackq(a, slx.nvar, legals.size(), tau, cols, qtmp);
    rmatrixqrunpackr(a, slx.nvar, legals.size(), r);
    // get rid of degenerate G
    int newdim = 0;
    for (;newdim < cols; ++newdim)
        if (abs(r[newdim][newdim]) < 1e-6)
            break;
    r0.setlength(newdim, newdim);
    r1.setlength(newdim, legals.size() - newdim);
    q.setlength(slx.nvar, newdim);
    for (int i = 0; i < slx.nvar; ++i)
        for (int j = 0; j < newdim; ++j)
            q[i][j] = qtmp[i][j];
    for (int i = 0; i < newdim; ++i) {
        for (int j = 0; j < newdim; ++j)
            r0[i][j] = r[i][j];
        for (int j = newdim; j < legals.size(); ++j)
            r1[i][j-newdim] = r[i][j];
    }
    r1lam1.setlength(newdim);
    for (int i = 0; i < newdim; ++i) {
        r1lam1[i] = 0;
        for (int j = newdim; j < legals.size(); ++j)
            r1lam1[i] += r1[i][j-newdim] * lambda[legals[j]];
    }
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r0, (ae_int_t)newdim, r1lam1, info, rep, lamp);
    for (int j = 0; j < newdim; ++j)
        lambda[legals[j]] += lamp[j];
    for (int j = newdim; j < legals.size(); ++j)
        lambda[legals[j]] = 0;
}

vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone *zone) {
    auto slx = NormalOpt(zone);
    cout << "orisize=" << slx.nvar + slx.ncon<<endl;
    double x[slx.nvar],oricon[slx.ncon];
    int sign;
    slx.initialize(x);
    auto lambda = augmented_lagrangian_method(slx);
    // do qr decomposition on sqrt(m^-1)G^T
    vector<int> legals;
    double *grads[slx.ncon], tmp;
    for (int i = 0; i < slx.ncon; ++i) {
        tmp = slx.constraint(&slx.tmp[0],i,sign);
        grads[i] = NULL;
        if (sign==1 && tmp>1e-6) continue;//sign==1:tmp>=0
        if (sign==-1 && tmp<-1e-6) continue;
        grads[i] = new double[slx.nvar];
        for (int j = 0; j < slx.nvar; ++j)
            grads[i][j]=0;
        slx.con_grad(&slx.tmp[0],i,1,grads[i]);
        legals.push_back(i);
    }
    real_2d_array a, q, r;
    real_1d_array sm_1;//sqrt(m^-1)
    precompute_derivative(a, q, r, lambda, sm_1, legals, grads, zone, slx);
    Tensor q_tn = arr2ten(q), r_tn = arr2ten(r);
    Tensor lam_tn = ptr2ten(&lambda[0], lambda.size());
    Tensor sm1_tn = ptr2ten(sm_1.getcontent(), sm_1.length());
    Tensor legals_tn = ptr2ten(&legals[0], legals.size());
    Tensor ans = ptr2ten(&slx.tmp[0], slx.nvar);
    for (int i = 0; i < slx.ncon; ++i) {
        delete [] grads[i];
    }
    return {ans.reshape({-1, 3}), q_tn, r_tn, lam_tn, sm1_tn, legals_tn};
}

void apply_inelastic_projection (ImpactZone *zone,
                                 const vector<Constraint*> &cons, bool verbose) {
    py::object func = py::module::import("collision_py").attr("apply_inelastic_projection");
    Tensor inp_xold, inp_w, inp_n;
    vector<Tensor> xolds(zone->nodes.size()), ws(zone->impacts.size()*4), ns(zone->impacts.size());
    for (int i = 0; i < zone->nodes.size(); ++i)
        xolds[i] = get_xold(zone->nodes[i]);
    for (int j = 0; j < zone->impacts.size(); ++j) {
        ns[j] = zone->impacts[j].n;
        for (int k = 0; k < 4; ++k)
            ws[j*4+k] = zone->impacts[j].w[k];
    }
    inp_xold = torch::stack(xolds);
    inp_w = torch::stack(ws);
    inp_n = torch::stack(ns);
    double *dw = inp_w.data<double>(), *dn = inp_n.data<double>();
    zone->w = vector<double>(dw, dw+zone->impacts.size()*4);
    zone->n = vector<double>(dn, dn+zone->impacts.size()*3);
    Tensor out_x = func(inp_xold, inp_w, inp_n, zone).cast<Tensor>();
    //Tensor out_x = apply_inelastic_projection_forward(inp_xold, inp_w, inp_n, zone)[0];
    for (int i = 0; i < zone->nodes.size(); ++i)
        zone->nodes[i]->x = out_x[i];
}

vector<Tensor> compute_derivative(real_1d_array &ans, ImpactZone *zone,
                        real_2d_array &q, real_2d_array &r, real_1d_array &sm_1, vector<int> &legals, 
                        real_1d_array &dldx,
                        vector<double> &lambda, bool verbose=false) {
    real_1d_array qtx, dz, dlam0, dlam, ana, dldw0, dldn0;
    int nvar = zone->nodes.size()*3;
    int ncon = zone->impacts.size();
    qtx.setlength(q.cols());
    ana.setlength(nvar);
    dldn0.setlength(ncon*3);
    dldw0.setlength(ncon*4);
    dz.setlength(nvar);
    dlam0.setlength(q.cols());
    dlam.setlength(ncon);
    for (int i = 0; i < nvar; ++i)
        ana[i] = dz[i] = 0;
    for (int i = 0; i < ncon*3; ++i) dldn0[i] = 0;
    for (int i = 0; i < ncon*4; ++i) dldw0[i] = 0;
    // qtx = qt * sqrt(m^-1) dldx
    for (int i = 0; i < q.cols(); ++i) {
        qtx[i] = 0;
        for (int j = 0; j < nvar; ++j)
            qtx[i] += q[j][i] * dldx[j] * sm_1[j];
    }
    // dz = sqrt(m^-1) (sqrt(m^-1) dldx - q * qtx)
    for (int i = 0; i < nvar; ++i) {
        dz[i] = dldx[i] * sm_1[i];
        for (int j = 0; j < q.cols(); ++j)
            dz[i] -= q[i][j] * qtx[j];
        dz[i] *= sm_1[i];
    }
    // dlam = R^-1 * qtx
    ae_int_t info;
    alglib::densesolverreport rep;
    cout << "orisize=" << nvar<<" "<<ncon<<" "<<nvar+ncon;
    cout << "  size=" << q.cols() << endl;
    rmatrixsolve(r, (ae_int_t)q.cols(), qtx, info, rep, dlam0);
// cout<<endl;
    for (int j = 0; j < ncon; ++j)
        dlam[j] = 0;
    for (int k = 0; k < q.cols(); ++k)
        dlam[legals[k]] = dlam0[k];
    //part1: dldq * dqdxt = M dz
    for (int i = 0; i < nvar; ++i)
        ana[i] += dz[i] / sm_1[i] / sm_1[i];
    //part2: dldg * dgdw * dwdxt
    for (int j = 0; j < ncon; ++j) {
        Impact &imp=zone->impacts[j];
        double *dldn = dldn0.getcontent() + j*3;
        for (int n = 0; n < 4; n++) {
            int i = find(imp.nodes[n], zone->nodes);
            double &dldw = dldw0[j*4+n];
            if (i != -1) {
                for (int k = 0; k < 3; ++k) {
                    //g=-w*n*x
                    dldw += (dlam[j]*ans[i*3+k]+lambda[j]*dz[i*3+k])*imp.n[k].item<double>();
    //part3: dldg * dgdn * dndxt
                    dldn[k] += imp.w[n].item<double>()*(dlam[j]*ans[i*3+k]+lambda[j]*dz[i*3+k]);
                }
            } else {
    //part4: dldh * (dhdw + dhdn)
                for (int k = 0; k < 3; ++k) {
                    dldw += (dlam[j] * imp.n[k] * imp.nodes[n]->x[k]).item<double>();
                    dldn[k] += (dlam[j] * imp.w[n] * imp.nodes[n]->x[k]).item<double>();
                }
            }
        }
    }
    Tensor grad_xold = torch::from_blob(ana.getcontent(), {nvar/3, 3}, TNOPT).clone();
    Tensor grad_w = torch::from_blob(dldw0.getcontent(), {ncon*4}, TNOPT).clone();
    Tensor grad_n = torch::from_blob(dldn0.getcontent(), {ncon, 3}, TNOPT).clone();
    // cout << grad_xold<<endl << grad_w.reshape({ncon,4})<<endl << grad_n<<endl << endl;
    delete zone;
    return {grad_xold, grad_w, grad_n};
}

vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone *zone) {
    real_2d_array q = ten2arr(q_tn), r = ten2arr(r_tn);
    real_1d_array sm_1 = ten1arr(sm1_tn), ans = ten1arr(ans_tn.reshape({-1})), dldx = ten1arr(dldx_tn.reshape({-1}));
    vector<double> lambda = ten2vec<double>(lam_tn);
    vector<int> legals = ten2vec<int>(legals_tn);
    // cout << dldx_tn<<endl << q_tn<<endl;
    return compute_derivative(ans, zone, q, r, sm_1, legals, dldx, lambda);
}

void NormalOpt::initialize (double *x) const {
    for (int n = 0; n < zone->nodes.size(); n++)
        set_subvec(x, n, zone->nodes[n]->x);
}

void NormalOpt::precompute (const double *x) const {
    for (int n = 0; n < zone->nodes.size(); n++)
        zone->nodes[n]->x = get_subvec(x, n);
}

double NormalOpt::objective (const double *x) const {
    double e = 0;
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        Tensor dx = node->x - get_xold(node);
        e = e + (inv_m*get_mass(node)*dot(dx, dx)/2).item<double>();
    }
    return e;
}

void NormalOpt::obj_grad (const double *x, double *grad) const {
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node *node = zone->nodes[n];
        Tensor dx = node->x - get_xold(node);
        set_subvec(grad, n, inv_m*get_mass(node)*dx);
    }
}

double NormalOpt::constraint (const double *x, int j, int &sign) const {
    sign = -1;
    double c = ::thickness.item<double>();
    const Impact &impact = zone->impacts[j];
    for (int n = 0; n < 4; n++)
        // c = c + (impact.w[n]*dot(impact.n, impact.nodes[n]->x)).item<double>();
    {
        double *dx = impact.nodes[n]->x.data<double>();
        for (int k = 0; k < 3; ++k)
            c -= zone->w[j*4+n]*zone->n[j*3+k]*dx[k];
    }
    return c;
}

void NormalOpt::con_grad (const double *x, int j, double factor,
                          double *grad) const {
    const Impact &impact = zone->impacts[j];
    for (int n = 0; n < 4; n++) {
        int i = find(impact.nodes[n], zone->nodes);
        if (i != -1)
            // add_subvec(grad, i, factor*impact.w[n]*impact.n);
            for (int k = 0; k < 3; ++k)
                grad[i*3+k] -= factor*zone->w[j*4+n]*zone->n[j*3+k];
    }
}

void NormalOpt::finalize (const double *x) {
    precompute(x);
    for (int i = 0; i < nvar; ++i)
        tmp[i] = x[i];
}

Tensor &get_xold (const Node *node) {
    pair<bool,int> mi = find_in_meshes(node);
    int ni = get_index(node, mi.first ? *::meshes : *::obs_meshes);
    return (mi.first ? ::xold : ::xold_obs)[ni];
}

}; //namespace CO


void collision_response (vector<Mesh*> &meshes, const vector<Constraint*> &cons,
                         const vector<Mesh*> &obs_meshes, bool verbose) {
    CO::collision_response(meshes, cons, obs_meshes, verbose);
}
