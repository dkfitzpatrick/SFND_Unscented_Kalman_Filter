#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF(const int n) :   // basically only handling default case.
n_x_(n),           // State dimension
n_aug_(n + 2),     // Augmented state dimension
lambda_(3 - n),    // Sigma point spreading parameter
is_initialized_(false),
use_laser_(true),  // ignore laser measurements - except during init
use_radar_(true),  // ignore radar measurements - except during init
std_a_(30),        // Process noise standard deviation longitudinal acceleration in m/s^2
std_yawdd_(30),    // Process noise standard deviation yaw acceleration in rad/s^2
std_laspx_(0.15),  // Don't Modify:  Laser measurement noise standard deviation position1 in m
std_laspy_(0.15),  // Don't Modify:  Laser measurement noise standard deviation position2 in m
std_radr_(0.3),    // Don't Modify:  Radar measurement noise standard deviation radius in m
std_radphi_(0.03), // Don't Modify:  Radar measurement noise standard deviation angle in rad
std_radrd_(0.3),   // Don't Modify:  Radar measurement noise standard deviation radius change in m/s
time_us_(0),
x_(n),             // initial state vector
P_(n, n),          // initial covariance matrix
Xsig_pred_(n, 2*(n + 2) + 1),
weights_(2*(n + 2) + 1),
h_laser_(2, n),
r_laser_(2, 2)
{
  /**
   * Hint: one or more values initialized above might be wildly off...
   */

  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_/(lambda_ + n_aug_);

  h_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  r_laser_ << std_laspx_*std_laspx_, 0,
             0, std_laspy_*std_laspy_;
}

UKF::~UKF() {
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_) {
    Init(meas_package);
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR ||
    !use_laser_  && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_)/1.0e6;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

void UKF::Init(const MeasurementPackage &meas_package) {
  P_ = MatrixXd::Identity(n_x_, n_x_);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    float rho = meas_package.raw_measurements_(0);
    float phi = meas_package.raw_measurements_(1);
    float rhodot = meas_package.raw_measurements_(2);
    float px = rho*cos(phi);
    float py = rho*sin(phi);
    float vx = rhodot*cos(phi);
    float vy = rhodot*sin(phi);

    x_ << px, py, sqrt(vx*vx + vy*vy), 0, 0;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    float px = meas_package.raw_measurements_(0);
    float py = meas_package.raw_measurements_(1);

    x_ << px, py, 0, 0, 0;
  } else {
    assert(0);
  }
}

void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = MatrixXd(2*n_aug_ + 1, n_aug_);
  ComputeAugmentedSigmaPoints(Xsig_aug);
  PredictSigmaPoints(Xsig_aug, delta_t);
  PredictSigmaMeanCovariance();
}

void UKF::ComputeAugmentedSigmaPoints(MatrixXd &Xsig_aug) {
  // set example state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  for (int i = n_x_; i < n_aug_; i++) {
    x_aug(i) = 0;
  }

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

  // calculate square root of P_aug
  MatrixXd A = P_aug.llt().matrixL();

  Xsig_aug.fill(0);
  // set first column of sigma point matrix
  Xsig_aug.col(0) = x_aug;
  // set remaining sigma points
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_)*A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*A.col(i);
  } 
}

void UKF::PredictSigmaPoints(MatrixXd &Xsig_aug, double delta_t) {

  for (int i = 0; i< 2*n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd*(sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd*(cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t*cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t*sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictSigmaMeanCovariance() {
  // weights_.fill(0.5/(lambda_ + n_aug_));
  // weights_(0) = lambda_/(lambda_ + n_aug_);

  // predict state mean
  // for (int i = 0; i < 2*n_aug_ + 1; ++i) {  // iterate over sigma points
  //   x_ = weights_(i)*Xsig_pred.col(i);
  // }

  x_ = Xsig_pred_*weights_;

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) >  M_PI) {
      x_diff(3) -= 2.0*M_PI;
    }
    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2.0*M_PI;
    }

    P_ += weights_(i)*x_diff*(x_diff.transpose());
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // Project Measurements

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double vel = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*vel;
    double v2 = sin(yaw)*vel;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                     // r
    Zsig(1,i) = atan2(p_y, p_x);                             // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2)/sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  // mean predicted measurement
  for (int i = 0; i < 2*n_aug_ + 1; ++i) {
    z_pred += weights_(i)*Zsig.col(i);
  }
//   z_pred = weights_*Zsig;

  // innovation covariance matrix S
  for (int i = 0; i < 2*n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i)*z_diff*(z_diff.transpose());
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radr_*std_radr_;

  S += R;

  // update component

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < 2*n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.0*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.0*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0*M_PI;

    Tc += weights_(i)*x_diff*z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc*S.inverse();

  // residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*(K.transpose());
}