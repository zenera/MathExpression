#include "MathExpression.h"
#include <iomanip>

using namespace me;

struct Test : public CustomFunction {
  double evaluate(double* args) override { return args[0] + args[1]; }
};

struct Test1 : public CustomFunction {
  double evaluate(double* args) override { return args[0] * args[1]; }
};

struct Rosenbrock : public CustomFunction {
  double evaluate(double* args) override {
    double x = args[0], y = args[1];
    return 100 * std::pow((x * x - y), 2) + std::pow(1 - y, 2);
  }
};

int main() {
#if 01
  CustomFunction::add<Test>("test");
  CustomFunction::add<Test1>("test1");
  CustomFunction::add<Rosenbrock>("rosen");

  MathExpression
  //
  // me("ceil(abs(a - (b-1)!*sin(18*pi/180*sqrt(a^2+b^2))))");
  // me("-5*((-2 - x)*-2) - (3*y)/(0+3) + 1");
  // me("7*a^7+6*a^6+5*a^5+4*a^4+3*a^3+2*a^2+1*a^1+0.1");
  // me("2^-3*8*a^(3*a-a-1) + 1");
#if 0
    me(R"( 2^2^(1+2)*2/2*(-a+3) + 1 + pow(a, 3+1-1) + 1*(a+1)!*2
           + 2*sin(pi/6) + a*4!*2 + a +b + log(e^-9) + 2*test(a, b)
           + test1(-2,5) )");
#endif
    //  me(R"( 3*sin(pi/2)*4 + pow(2, 3) )");
    // me("2^a^(2+1) + 1*a");
    // me("a*pow(2, 3) + 2*sin(rad(90))");
    // me("(a+1)!*(a-3)*3 + 2*npr(4,2)");
    // me("(2*(a-b)*3!-3*2*(-a+b))");
    // me(R"(a = b+7; 2*test(4,6) + a + b*2)");
    // me(R"(a = b-1; c = 2*(3 + 4*(2+3*(a+1)));)");
    me("x = 2; y= 3; r=rosen(x,y); z = r+1;");
  // double result = me.evaluate(0.0, 2.0, 3.0, 0.0);
  double result = me.evaluate();
  me.outVars();
  std::cout << "Result: " << result << '\n' << std::endl;
#endif

#if 0
  MathExpression mex("2*(2!)^3/4*sin(pi*x/ 180)");
  for (int x = 0; x <= 360; x += 30) {
    double result = mex.evaluate(x);
    if (std::abs(result) < 1e-12) result = 0;
    std::cout << std::fixed << std::setw(8) << std::setprecision(3)
              << "Result: " << result << "\t<= ";
    mex.outVars();
  }
#endif

  return 0;
}
