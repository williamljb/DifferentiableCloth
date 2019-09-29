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

#ifndef COLLISION_HPP
#define COLLISION_HPP

#include "cloth.hpp"
#include "constraint.hpp"
using torch::Tensor;

void collision_response (std::vector<Mesh*> &meshes,
                         const std::vector<Constraint*> &cons,
                         const std::vector<Mesh*> &obs_meshes,bool verbose=false);
namespace CO {
struct Impact {
    enum Type {VF, EE} type;
    Tensor t;
    Node *nodes[4];
    Tensor w[4];
    Tensor n;
    Impact () {}
    Impact (Type type, const Node *n0, const Node *n1, const Node *n2,
            const Node *n3): type(type) {
        nodes[0] = (Node*)n0;
        nodes[1] = (Node*)n1;
        nodes[2] = (Node*)n2;
        nodes[3] = (Node*)n3;
    }
};

struct ImpactZone {
    vector<Node*> nodes;
    vector<Impact> impacts;
    vector<double> w, n;
    bool active;
};
} //namespace CO

#endif
