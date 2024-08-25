/*
 * Copyright (c) 2011-2021, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <dart/dart.hpp>
#include <dart/utils/urdf/urdf.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dart {
namespace dynamics {

struct Anchor
{
  std::vector<dart::dynamics::BodyNode*> bodynodes;
  std::vector<Eigen::Vector3d> local_positions;
  std::vector<double> weights;
  int num_related_bodies;
  dart::dynamics::BodyNode* explicit_bodynode;

  Anchor(
      std::vector<dart::dynamics::BodyNode*> bns,
      std::vector<Eigen::Vector3d> lps,
      std::vector<double> ws);
  Eigen::Vector3d GetPoint();
};
class Muscle
{
private:
  // For fatigue
  double mActiveUnits;
  double mRestingUnits;
  double mFatigueUnits;
  double pen_angle;
  bool selected;
  bool mUseVelocityForce;
  std::string name;
  double f0_original;
  double f0;
  double v_m;
  double l_m0;
  double l_m;
  double l_t0;
  double l_mt0, l_mt;
  double activation;

public:
  Muscle(
      std::string _name,
      double f0,
      double lm0,
      double lt0,
      double pen_angle,
      double lmax,
      double type1_fraction,
      bool useVelocityForce = false);
    void setActivation(double a) {activation = a;}
  void AddAnchor(
      const dart::dynamics::SkeletonPtr& skel,
      dart::dynamics::BodyNode* bn,
      const Eigen::Vector3d& glob_pos);
  void AddAnchor(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& glob_pos);
  const std::vector<Anchor*>& GetAnchors()
  {
    return mAnchors;
  }
  bool Update();
  void UpdateVelocities();
  void ApplyForceToBody();

  double GetForce();
  double GetActiveForce()
  {
    return Getf_A() * activation;
  };

  double Getf_A();
  double Getf_p();
  double Getl_mt();
  double GetNormalizedLength();
  double GetRecommendedMinLength();

  double GetLength()
  {
    double length = 0.0;
    for (int i = 1; i < (int)mAnchors.size(); i++)
      length
          += (mCachedAnchorPositions[i] - mCachedAnchorPositions[i - 1]).norm();
    return length;
  }
  double GetVelocity()
  {
    return v_m;
  }

  std::vector<std::vector<double>> GetGraphData();

  void SetMuscle();
  const std::vector<Anchor*>& GetAnchors() const
  {
    return mAnchors;
  }
  void set_l_mt_max(double l_max)
  {
    l_mt_max = l_max;
  }

  Eigen::MatrixXd GetJacobianTranspose();
  Eigen::MatrixXd GetReducedJacobianTranspose();

  std::pair<Eigen::VectorXd, Eigen::VectorXd> GetForceJacobianAndPassive();

  int GetNumRelatedDofs()
  {
    return num_related_dofs;
  };

  Eigen::VectorXd GetRelatedJtA();
  Eigen::VectorXd GetRelatedJtp();

  std::vector<dart::dynamics::Joint*> GetRelatedJoints();
  std::vector<dart::dynamics::BodyNode*> GetRelatedBodyNodes();
  void ComputeJacobians();
  Eigen::VectorXd Getdl_dtheta();

  double GetLengthRatio()
  {
    return length / l_mt0;
  };
  std::string GetName()
  {
    return name;
  }

  void setFatigueUnits(double units)
  {
    mFatigueUnits = units;
  }

public:
  std::vector<Anchor*> mAnchors;
  int num_related_dofs;
  int getNumRelatedDofs()
  {
    return num_related_dofs;
  }

  std::vector<int> original_related_dof_indices;
  std::vector<int> related_dof_indices;

  std::vector<Eigen::Vector3d> mCachedAnchorPositions;
  std::vector<Eigen::Vector3d> mCachedAnchorVelocities;
  std::vector<Eigen::MatrixXd> mCachedJs;

  // New
  void change_f(double ratio)
  {
    f0 = f0_original * ratio;
  }
  void change_l(double ratio)
  {
    l_mt0 = l_mt0_original * ratio;
  }

  double ratio_f()
  {
    return f0 / f0_original;
  }
  double ratio_l()
  {
    return l_mt0 / l_mt0_original;
  }

  // Dynamics
  double g(double _l_m);
  double g_t(double e_t);

  double g_pl(double _l_m);
  double g_al(double _l_m);
  double g_av(double _l_m);

  double length;

  double l_mt0_original;

  double f_min;
  double l_min;

  double f_toe, k_toe, k_lin, e_toe, e_t0; // For g_t
  double k_pe, e_mo;                       // For g_pl
  double gamma;                            // For g_al
  double l_mt_max;

  Eigen::VectorXd related_vec;
  Eigen::VectorXd GetRelatedVec()
  {
    return related_vec;
  }
  double GetForce0()
  {
    return f0;
  }

  double GetMass();
  double GetBHAR04_EnergyRate();
  double type1_fraction;

  double GetType1_Fraction()
  {
    return type1_fraction;
  }
  double Getdl_velocity();

  double g_fatigue(); // fatigue
  void fatigue_update(double dt = 1.0 / 480);
  void fatigue_reset();
};

class Muscles
{
  std::vector<Muscle*> mMuscles;
  dart::dynamics::SkeletonPtr mSkel;
  int mNumMuscleRelatedDof;

public:
  Muscles(dart::dynamics::SkeletonPtr skel)
  {
    mSkel = skel;
    mMuscles.clear();
    mNumMuscleRelatedDof = 0;
  }
  void addMuscles(
      std::string name,
      std::vector<double> muscle_properties,
      bool useVelocityForce,
      std::vector<std::pair<std::string, Eigen::Vector3d>> anchors)
  {
    Muscle* muscle_elem = new Muscle(
        name,
        muscle_properties[0],
        muscle_properties[1],
        muscle_properties[2],
        muscle_properties[3],
        muscle_properties[4],
        muscle_properties[5],
        useVelocityForce);

    for (int i = 0; i < (int)anchors.size(); i++)
    {
      auto anchor = anchors[i];
      if (i == 0 || i == (int)anchors.size() - 1)
        muscle_elem->AddAnchor(mSkel->getBodyNode(anchor.first), anchor.second);
      else
        muscle_elem->AddAnchor(
            mSkel, mSkel->getBodyNode(anchor.first), anchor.second);
    }
    muscle_elem->SetMuscle();
    mNumMuscleRelatedDof += muscle_elem->GetNumRelatedDofs();
    if (muscle_elem->GetNumRelatedDofs() > 0)
      mMuscles.push_back(muscle_elem);
  }

  void update()
  {
    for (auto m : mMuscles)
      m->Update();
  }
  void applyForceToBody()
  {
    for (auto m : mMuscles)
      m->ApplyForceToBody();
  }

  int getNumMuscleRelatedDofs()
  {
    return mNumMuscleRelatedDof;
  }

  int getNumMuscles()
  {
    return (int)mMuscles.size();
  }

  std::vector<std::vector<Eigen::Vector3d>> getMusclePositions()
  {
    std::vector<std::vector<Eigen::Vector3d>> anchor_positions;
    for (auto m : mMuscles)
    {
      std::vector<Eigen::Vector3d> muscle_anchor_positions;
      for (auto a : m->GetAnchors())
      {
        muscle_anchor_positions.push_back(a->GetPoint());
      }
      anchor_positions.push_back(muscle_anchor_positions);
    }
    return anchor_positions;
  }

  dart::dynamics::SkeletonPtr getSkel()
  {
    return mSkel;
  }

  // JtA_reduced, JtP, JtA
  std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::MatrixXf> getMuscleTuples()
  {

    int n = mSkel->getNumDofs();
    int m = mMuscles.size();

    Eigen::VectorXd res_JtA_reduced = Eigen::VectorXd::Zero(mNumMuscleRelatedDof);

    Eigen::VectorXd JtP = Eigen::VectorXd::Zero(n);
    Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n, m);
    
    if (mMuscles.size() == 0)
    {
      std::cout << "Muscles::getMuscleTuple()\tNo Muscles" << std::endl;
      exit(-1);
    }

    int i = 0;
    int idx = 0;
    for (auto m : mMuscles)
    {
      m->Update();
      m->related_vec.setZero();
      Eigen::MatrixXd Jt_reduced = m->GetReducedJacobianTranspose();
      auto Ap = m->GetForceJacobianAndPassive();
      Eigen::VectorXd JtA_reduced = Jt_reduced * Ap.first;
      Eigen::VectorXd JtP_reduced = Jt_reduced * Ap.second;
      for (int j = 0; j < m->getNumRelatedDofs(); j++)
      {
        JtP[m->related_dof_indices[j]] += JtP_reduced[j];
        JtA(m->related_dof_indices[j], i) = JtA_reduced[j];
        m->related_vec[m->related_dof_indices[j]] = JtA_reduced[j];
      }
      res_JtA_reduced.segment(idx, JtA_reduced.rows()) = JtA_reduced;
      idx += JtA_reduced.rows();
      i++;
    }

    return std::make_tuple(
        res_JtA_reduced.cast<float>(),
        JtP.tail(mSkel->getNumDofs() - mSkel->getRootJoint()->getNumDofs())
            .cast<float>(),
        JtA.block(
               mSkel->getRootJoint()->getNumDofs(),
               0,
               JtA.rows() - mSkel->getRootJoint()->getNumDofs(),
               JtA.cols())
            .cast<float>());
    // Test For Reduced Jacobian
    // for (auto m : mMuscles)
    // {
    //     Eigen::VectorXd related_vec_backup = m->related_vec;
    //     m->Update();
    //     Eigen::MatrixXd Jt = m->GetJacobianTranspose();
    //     auto Ap = m->GetForceJacobianAndPassive();
    //     Eigen::VectorXd JtA = Jt * Ap.first;
    //     m->related_vec.setZero();
    //     for (int i = 0; i < JtA.rows(); i++)
    //         if (JtA[i] > 1E-6)
    //             m->related_vec[i] = 1;
    //         else if (JtA[i] < -1E-6)
    //             m->related_vec[i] = -1;

    //     if ((related_vec_backup - m->related_vec).norm() < 1E-6)
    //         std::cout << "DIFFERENT MUSCLE " << m->name << std::endl;
    // }
    
  }
  void setActivations(Eigen::VectorXd a)
  {
    int i = 0;
    for (auto m : mMuscles)
    {
      m->setActivation(a[i]);
      i++;
    }
  }
};

} // namespace dynamics
} // namespace dart
