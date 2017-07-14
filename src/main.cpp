#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(std::vector<double> xvals, std::vector<double> yvals,
                        unsigned int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  Eigen::VectorXd y(yvals.size());

  for (unsigned int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
    y(i) = yvals[i];
  }

  for (unsigned int j = 0; j < xvals.size(); j++) {
    for (unsigned int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals[j];
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(y);
  return result;
}

double get_y(Eigen::VectorXd coeffs, double x) {
  return coeffs[0]  + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x;
}

std::pair<double, double> globalToVehicle(double gx, double gy, double vx, double vy, double psi) {
  double dx = gx - vx;
  double dy = gy - vy;
  double x = dx * std::cos(psi) + dy * std::sin(psi);
  double y = - dx * std::sin(psi) + dy * std::cos(psi);
  return std::make_pair(x, y);
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double angle = j[1]["steering_angle"];
          double acc = j[1]["throttle"];

          vector<double> vptsx;
          vector<double> vptsy;
          for (unsigned i = 0; i < ptsx.size(); ++i) {
            auto p = globalToVehicle(ptsx[i], ptsy[i], px, py, psi);
            vptsx.push_back(p.first);
            vptsy.push_back(p.second);
          }

          Eigen::VectorXd coeffs = polyfit(vptsx, vptsy, 3);
          double lattency = 0.1;
          double predicted_x = px + v * std::cos(psi) * lattency;
          double predicted_y = py + v * std::sin(psi) * lattency;
          auto local_predicted = globalToVehicle(predicted_x, predicted_y, px, py, psi);
          double cte = polyeval(coeffs, local_predicted.first) - local_predicted.second;
          std::cout << "CTE = " << cte << std::endl;
          double slope = std::atan(coeffs[1] + 2. * coeffs[2] * local_predicted.first + 3. * coeffs[3] * std::pow(local_predicted.first, 2));
          double epsi = - slope;
          Eigen::VectorXd state(6);
          state << local_predicted.first, local_predicted.second, -angle * v * lattency / 2.67, v + acc * lattency, cte, epsi;

          vector<double> solution = mpc.Solve(state, coeffs);

          double steer_value = solution[0] / deg2rad(25);
          steer_value = std::min(steer_value, 1.);
          steer_value = std::max(steer_value, -1.);
          double throttle_value = solution[1];
          throttle_value = std::min(throttle_value, 1.);
          throttle_value = std::max(throttle_value, -1.);

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = - steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          for (size_t i = 0; i < mpc.predicted_x_.size(); ++i) {
            auto p = globalToVehicle(mpc.predicted_x_[i], mpc.predicted_y_[i], px, py, psi);
            mpc_x_vals.push_back(p.first);
            mpc_y_vals.push_back(p.second);
          }
          for (size_t i = 0; i < vptsx.size(); ++i) {
            std::cout << vptsx[i] << ", ";
          }
          std::cout << std::endl;
          for (size_t i = 0; i < vptsx.size(); ++i) {
            std::cout << vptsy[i] << ", ";
          }
          std::cout << std::endl;
          std::cout << coeffs[0] << ", " << coeffs[1] << ", " << coeffs[2] << ", " << coeffs[3] << std::endl;
          mpc_x_vals = mpc.predicted_x_;
          mpc_y_vals = mpc.predicted_y_;

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          for (size_t i = 0; i < vptsx.size(); ++i) {
            double gy = get_y(coeffs, vptsx[i]);
            //auto p = globalToVehicle(ptsx[i], ptsy[i], px, py, psi);
            next_x_vals.push_back(vptsx[i]);
            next_y_vals.push_back(gy);
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
