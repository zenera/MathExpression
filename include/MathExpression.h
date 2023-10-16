/*
The MIT License (MIT)

Copyright (c) 2023 Zenera (zenera@naver.com)
https://github.com/zenera/MathExpression

MathExpression: simple recursive descent parser for math expressions.
  - for more information:
    https://en.wikipedia.org/wiki/Recursive_descent_parser
    https://github.com/ArashPartow/math-parser-benchmark-project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef MATH_EXPRESSION_H
#define MATH_EXPRESSION_H

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace me {

enum EType {
  ET_NONE,  // No type
  ET_NUM,   // Constant numbers
  ET_VAR,   // Variables
  ET_FUN,   // Built-in functions
  ET_CFUN,  // Custom functions
  ET_FAC,   // '!' : Factorial
  ET_ADD,   // '+'
  ET_SUB,   // '-'
  ET_MUL,   // '*'
  ET_DIV,   // '/'
  ET_MOD,   // '%'
  ET_POW,   // '^'
  ET_OPEN,  // '('
  ET_CLOSE, // ')'
  ET_SEP,   // ',' : Function argument list
  ET_SET,   // '=' : Assign expression
  ET_TRM    // ';' : Terminate expression
};

enum EFunction {
  EF_NONE,
  EF_ABS,
  EF_ACOS,
  EF_ASIN,
  EF_ATAN,
  EF_ATAN2,
  EF_CBRT,
  EF_CEIL,
  EF_COS,
  EF_COSH,
  EF_DEG,
  EF_ERF,
  EF_ERFC,
  EF_EXP,
  EF_FLOOR,
  EF_HYPOT,
  EF_LOG,
  EF_LOG2,
  EF_LOG10,
  EF_MAX,
  EF_MIN,
  EF_NCR,
  EF_NPR,
  EF_POW,
  EF_RAD,
  EF_ROUND,
  EF_SIN,
  EF_SINH,
  EF_SQRT,
  EF_TAN,
  EF_TANH,
  EF_TRUNC
};

struct TreeNode;
struct Expression;
using UTreeNode = std::unique_ptr<TreeNode>;
using SExpression = std::shared_ptr<Expression>;

// Built-in functions.
std::map<std::string, EFunction> FunMap{
  {"abs", EF_ABS},     {"acos", EF_ACOS},   {"asin", EF_ASIN},
  {"atan", EF_ATAN},   {"atan2", EF_ATAN2}, {"cbrt", EF_CBRT},
  {"ceil", EF_CEIL},   {"cos", EF_COS},     {"cosh", EF_COSH},
  {"deg", EF_DEG},     {"erf", EF_ERF},     {"erfc", EF_ERFC},
  {"exp", EF_EXP},     {"floor", EF_FLOOR}, {"hypot", EF_HYPOT},
  {"log", EF_LOG},     {"log2", EF_LOG2},   {"log10", EF_LOG10},
  {"max", EF_MAX},     {"min", EF_MIN},     {"ncr", EF_NCR},
  {"npr", EF_NPR},     {"pow", EF_POW},     {"rad", EF_RAD},
  {"round", EF_ROUND}, {"sin", EF_SIN},     {"sinh", EF_SINH},
  {"sqrt", EF_SQRT},   {"tan", EF_TAN},     {"tanh", EF_TANH},
  {"trunc", EF_TRUNC}};

std::vector<double> Vars;          // Var(variables) storage
std::vector<std::string> VarNames; // unique Var names

struct Token {
  Token() = default;
  Token(EType t, double v = 0, EFunction fun = EF_NONE)
    : exprType(t), value{v}, funType{fun} {}
  Token(SExpression f, EType t = ET_CFUN)
    : function(std::move(f)), exprType(t) {}
  SExpression function;
  double value{0};
  EType exprType{ET_NONE};
  EFunction funType{EF_NONE};
};

struct TreeNode {
  TreeNode() = default;

  TreeNode(SExpression op, UTreeNode child = {}) : operation{std::move(op)} {
    if (child) children.push_back(std::move(child));
  }

  template <typename... TArgs>
  TreeNode(SExpression op, UTreeNode child, TArgs&&... args)
    : operation{std::move(op)} {
    children.push_back(std::move(child));
    (children.push_back(std::move(args)), ...);
  }

  TreeNode(SExpression op, std::vector<UTreeNode> cv)
    : operation{std::move(op)}, children(std::move(cv)) {}

  SExpression operation;
  std::vector<UTreeNode> children;
};

struct Expression {
  virtual double evaluate(double* args) = 0;
};

// Singleton function factory for custom function classes.
class CustomFunction : public Expression {
  using TFunctionMap = std::map<std::string, SExpression>;
  using UFunctionMAp = std::unique_ptr<TFunctionMap>;

public:
  CustomFunction() : m_map{std::make_unique<TFunctionMap>()} {}

  static SExpression get(const std::string& name) {
    auto it = instance().getMap()->find(name);
    if (it == instance().getMap()->end()) return nullptr;
    return it->second;
  }

  template <typename TF, typename... TArgs>
  static SExpression add(const std::string& name, TArgs&&... args) {
    auto it = instance().getMap()->find(name);
    if (it != instance().getMap()->end()) return nullptr;
    SExpression pe = instance().create<TF>(std::forward<TArgs>(args)...);
    instance().getMap()->insert(std::make_pair(name, pe));
    return pe;
  }

protected:
  virtual double evaluate(double* args) override { return 0; }

  static CustomFunction& instance() {
    static CustomFunction instance;
    return instance;
  }

  template <typename TF, typename... TArgs>
  SExpression create(TArgs&&... args) {
    return std::make_shared<TF>(std::forward<TArgs>(args)...);
  }

  UFunctionMAp& getMap() { return m_map; }

private:
  UFunctionMAp m_map;
};

class Num : public Expression {
public:
  Num(double value) : m_value(value) {}
  double evaluate(double* args) override { return m_value; }

private:
  double m_value;
};

// Note: Var uses global storage.
class Var : public Expression {
public:
  Var(int id, bool child = false) : m_id(id), m_child{child} {}
  double evaluate(double* args) override {
    if (m_child) Vars[m_id] = args[0];
    return Vars[m_id];
  }

private:
  int m_id;
  bool m_child; // has child node?
};

struct Fac : public Expression {
  double evaluate(double* args) override {
    double nd = args[0];
    if (std::isnan(nd)) return nd;
    unsigned n = unsigned(nd);
    unsigned result = 1;
    for (unsigned i = 1; i <= n; ++i) result *= i;
    return double(result);
  }
};

struct Ncr : public Expression {
  double evaluate(double* args) override {
    double nd = args[0];
    if (std::isnan(nd)) return nd;
    unsigned n = unsigned(nd);
    nd = args[1];
    if (std::isnan(nd)) return nd;
    unsigned r = unsigned(nd);
    if (r > n / 2) r = n - r;
    unsigned result = 1;
    for (unsigned i = 1; i <= r; ++i) {
      result *= n - r + i;
      result /= i;
    }
    return double(result);
  }
};

struct Npr : public Expression {
  double evaluate(double* args) override {
    return Ncr().evaluate(args) * Fac().evaluate(&args[1]);
  }
};

struct Neg : public Expression {
  double evaluate(double* args) override { return -1 * args[0]; }
};

struct Add : public Expression {
  double evaluate(double* args) override { return args[0] + args[1]; }
};

struct Sub : public Expression {
  double evaluate(double* args) override { return args[0] - args[1]; }
};

struct Mul : public Expression {
  double evaluate(double* args) override { return args[0] * args[1]; }
};

struct Div : public Expression {
  double evaluate(double* args) override { return args[0] / args[1]; }
};

struct Mod : public Expression {
  double evaluate(double* args) override { return std::fmod(args[0], args[1]); }
};

struct Abs : public Expression {
  double evaluate(double* args) override { return std::abs(args[0]); }
};

struct Acos : public Expression {
  double evaluate(double* args) override { return std::acos(args[0]); }
};

struct Asin : public Expression {
  double evaluate(double* args) override { return std::asin(args[0]); }
};

struct Atan : public Expression {
  double evaluate(double* args) override { return std::atan(args[0]); }
};

struct Atan2 : public Expression {
  double evaluate(double* args) override {
    return std::atan2(args[0], args[1]);
  }
};

struct Cbrt : public Expression {
  double evaluate(double* args) override { return std::cbrt(args[0]); }
};

struct Ceil : public Expression {
  double evaluate(double* args) override { return std::ceil(args[0]); }
};

struct Cos : public Expression {
  double evaluate(double* args) override { return std::cos(args[0]); }
};

struct Cosh : public Expression {
  double evaluate(double* args) override { return std::cosh(args[0]); }
};

struct Deg : public Expression {
  double evaluate(double* args) override { return args[0] * 180.0 / M_PI; }
};

struct Erf : public Expression {
  double evaluate(double* args) override { return std::erf(args[0]); }
};

struct Erfc : public Expression {
  double evaluate(double* args) override { return std::erfc(args[0]); }
};

struct Exp : public Expression {
  double evaluate(double* args) override { return std::exp(args[0]); }
};

struct Floor : public Expression {
  double evaluate(double* args) override { return std::floor(args[0]); }
};

struct Hypot : public Expression {
  double evaluate(double* args) override {
    return std::hypot(args[0], args[1]);
  }
};

struct Log : public Expression {
  double evaluate(double* args) override { return std::log(args[0]); }
};

struct Log2 : public Expression {
  double evaluate(double* args) override { return std::log2(args[0]); }
};

struct Log10 : public Expression {
  double evaluate(double* args) override { return std::log10(args[0]); }
};

struct Max : public Expression {
  double evaluate(double* args) override { return std::max(args[0], args[1]); }
};

struct Min : public Expression {
  double evaluate(double* args) override { return std::min(args[0], args[1]); }
};

struct Pow : public Expression {
  double evaluate(double* args) override { return std::pow(args[0], args[1]); }
};

struct Rad : public Expression {
  double evaluate(double* args) override { return args[0] * M_PI / 180.0; }
};

struct Round : public Expression {
  double evaluate(double* args) override { return std::round(args[0]); }
};

struct Sin : public Expression {
  double evaluate(double* args) override { return std::sin(args[0]); }
};

struct Sinh : public Expression {
  double evaluate(double* args) override { return std::sinh(args[0]); }
};

struct Sqrt : public Expression {
  double evaluate(double* args) override { return std::sqrt(args[0]); }
};

struct Tan : public Expression {
  double evaluate(double* args) override { return std::tan(args[0]); }
};

struct Tanh : public Expression {
  double evaluate(double* args) override { return std::tanh(args[0]); }
};

struct Trunc : public Expression {
  double evaluate(double* args) override { return std::round(args[0]); }
};

class MathExpression {
public:
  MathExpression(const char* expr) : m_exprStr(expr) { compile(); }

  template <typename... TAs>
  void setVars(TAs... args) {
    int n_args = sizeof...(args), n_vars = sizeVars();
    if (n_args < n_vars) {
      std::cerr << "Warning: setVars() - too few input values. Variable count: "
                << n_vars << std::endl;
    } else if (n_args > n_vars) { // Allow extra values.
      for (int i = 0; i < n_args - n_vars; ++i) addVar("(unused)");
    }
    int i = 0;
    ((Vars[i++] = args), ...);
  }

  void outVars() {
    std::cout << "Vars: ";
    for (int i = 0; i < sizeVars(); ++i) {
      auto end_str = i < sizeVars() - 1 ? ", " : "\n";
      std::cout << VarNames[i] << " = " << Vars[i] << end_str;
    }
  }

  // Use preset variables.
  template <typename... TAs>
  double evaluate(double arg0, TAs&&... args) {
    setVars(arg0, std::forward<TAs>(args)...);
    return evaluateAll();
  }

  // Don't use preset variables.
  double evaluate() { return evaluateAll(); }

  int sizeVars() { return VarNames.size(); }

private:
  double evaluateAll() {
    double result;
    for (auto& expr : m_expressions) result = evaluate(expr.get());
    return result;
  }

  double evaluate(TreeNode* root) {
    int n_args = root->children.size();
    if (!n_args) {
      double arg = 0;
      return root->operation->evaluate(&arg);
    }

    double args[n_args]; // Variable Lenght Array(VLA) is not C++ standard.
    // double* args = new double[n_args]; // dirty part
    for (int i = 0; i < n_args; ++i)
      args[i] = evaluate(root->children[i].get());
    double result = root->operation->evaluate(args);
    // delete[] args; // dirty part
    return result;
  }

  int indexVar(const std::string& name) {
    for (int i = 0; i < sizeVars(); ++i)
      if (VarNames[i] == name) return i;
    return -1; // Not found.
  }

  void addVar(const std::string& name) {
    VarNames.push_back(name);
    Vars.push_back(std::nan(""));
  }

  // Simple optimization for constant Expressions.
  void optimize(UTreeNode* node) {
    if (!(*node).get() || !(*node)->children.size()) return;
    double result = evaluate((*node).get());
    if (!std::isnan(result)) {
      TreeNode tn = {std::make_shared<Num>(result)};
      *node = std::make_unique<TreeNode>(std::move(tn));
      // std::cout << "\nOptimized: " << result << std::endl;
    }
  }

  bool isName(char c) { return std::isalpha(c) || std::isdigit(c) || c == '_'; }

  Token nextToken() {
    auto& expr = m_exprStr;
    if (!expr) return {};
    for (char* end{}; *expr;) {
      if (std::isdigit(expr[0]) || expr[0] == '.') {
        double value = std::strtod(expr, &end);
        std::cout << value;
        expr = end;
        return {ET_NUM, value};
      } else {
        if (std::isalpha(expr[0])) {
          const char* start = expr++;
          while (isName(expr[0])) ++expr;
          auto name = std::string(start, expr - start);
          std::cout << name;
          if (name == "e") return {ET_NUM, std::exp(1.0)};
          else if (name == "pi") return {ET_NUM, M_PI};
          else if (auto cf = CustomFunction::get(name); cf) {
            return {cf};
          } else if (auto it = FunMap.find(name); it != FunMap.end())
            return {ET_FUN, 0, it->second};
          else {
            int id = indexVar(name);
            if (id < 0) {
              id = sizeVars();
              addVar(name);
            }
            return {ET_VAR, double(id)};
          }
        } else {
          if (!std::isspace(expr[0])) std::cout << expr[0];
          switch (expr++[0]) {
            case '!': return {ET_FAC};
            case '+': return {ET_ADD};
            case '-': return {ET_SUB};
            case '*': return {ET_MUL};
            case '/': return {ET_DIV};
            case '%': return {ET_MOD};
            case '^': return {ET_POW};
            case '(': return {ET_OPEN};
            case ')': return {ET_CLOSE};
            case '=': return {ET_SET};
            case ',': return {ET_SEP};
            case ';': return {ET_TRM};
            default: break;
          }
        }
      }
    }
    return {};
  }

  bool accept(EType type) { return (m_token.exprType == type); }

  bool expect(EType type) {
    if (m_token.exprType == type) return true;
    std::cerr << "Error: expect() - unexpected type." << std::endl;
    return false;
  }

  // <factor> : {'-' | '+'} <base> {{'^'} {'-' | '+'} <base> {'!'}}
  // <base>   : <constant> | <variable> {'=' <expression> ';'} |
  //            <function> '(' <list> ')' | '(' <expression> ')'
  UTreeNode factor() {
    UTreeNode node;
    if (accept(ET_NONE)) return {};
    else if (accept(ET_ADD) || accept(ET_SUB)) {
      auto type = m_token.exprType;
      m_token = nextToken();
      node = factor();
      if (type == ET_SUB) {
        TreeNode tn = {std::make_shared<Neg>(), std::move(node)};
        node = std::make_unique<TreeNode>(std::move(tn));
      }
    } else if (accept(ET_NUM)) {
      TreeNode tn = {std::make_shared<Num>(m_token.value)};
      node = std::make_unique<TreeNode>(std::move(tn));
      m_token = nextToken();
    } else if (accept(ET_VAR)) {
      int id = int(m_token.value);
      TreeNode tn = {std::make_shared<Var>(id)};
      node = std::make_unique<TreeNode>(std::move(tn));
      m_token = nextToken();
      if (accept(ET_SET)) {
        m_token = nextToken();
        TreeNode tn = {std::make_shared<Var>(id, true),
                       std::move(expression())};
        node = std::make_unique<TreeNode>(std::move(tn));
        m_expressions.push_back(std::move(node));
        expect(ET_TRM);
        m_token = nextToken();
        node = expression();
      }
    } else if (accept(ET_CFUN)) {
      auto custom_function = m_token.function;
      m_token = nextToken();
      node = factor();
      // To-do: Function arity checking is necessary here.
      // m_args.size() == the number of arguments for function.evaluate().
      TreeNode tn = {custom_function, std::move(m_args)};
      node = std::make_unique<TreeNode>(std::move(tn));
    } else if (accept(ET_FUN)) {
      TreeNode tn;
      auto fun_type = m_token.funType;
      m_token = nextToken();
      node = factor();
      switch (fun_type) {
        case EF_ABS: tn = {std::make_shared<Abs>(), std::move(node)}; break;
        case EF_ACOS: tn = {std::make_shared<Acos>(), std::move(node)}; break;
        case EF_ASIN: tn = {std::make_shared<Asin>(), std::move(node)}; break;
        case EF_ATAN: tn = {std::make_shared<Atan>(), std::move(node)}; break;
        case EF_ATAN2:
          tn = {std::make_shared<Atan2>(), std::move(m_args)};
          break;
        case EF_CBRT: tn = {std::make_shared<Cbrt>(), std::move(node)}; break;
        case EF_CEIL: tn = {std::make_shared<Ceil>(), std::move(node)}; break;
        case EF_COS: tn = {std::make_shared<Cos>(), std::move(node)}; break;
        case EF_COSH: tn = {std::make_shared<Cosh>(), std::move(node)}; break;
        case EF_DEG: tn = {std::make_shared<Deg>(), std::move(node)}; break;
        case EF_ERF: tn = {std::make_shared<Erf>(), std::move(node)}; break;
        case EF_ERFC: tn = {std::make_shared<Erfc>(), std::move(node)}; break;
        case EF_EXP: tn = {std::make_shared<Exp>(), std::move(node)}; break;
        case EF_FLOOR: tn = {std::make_shared<Floor>(), std::move(node)}; break;
        case EF_HYPOT:
          tn = {std::make_shared<Hypot>(), std::move(m_args)};
          break;
        case EF_LOG: tn = {std::make_shared<Log>(), std::move(node)}; break;
        case EF_LOG2: tn = {std::make_shared<Log2>(), std::move(node)}; break;
        case EF_LOG10: tn = {std::make_shared<Log10>(), std::move(node)}; break;
        case EF_MAX: tn = {std::make_shared<Max>(), std::move(m_args)}; break;
        case EF_MIN: tn = {std::make_shared<Min>(), std::move(m_args)}; break;
        case EF_NCR: tn = {std::make_shared<Ncr>(), std::move(m_args)}; break;
        case EF_NPR: tn = {std::make_shared<Npr>(), std::move(m_args)}; break;
        case EF_POW: tn = {std::make_shared<Pow>(), std::move(m_args)}; break;
        case EF_RAD: tn = {std::make_shared<Rad>(), std::move(node)}; break;
        case EF_ROUND: tn = {std::make_shared<Round>(), std::move(node)}; break;
        case EF_SIN: tn = {std::make_shared<Sin>(), std::move(node)}; break;
        case EF_SINH: tn = {std::make_shared<Sinh>(), std::move(node)}; break;
        case EF_SQRT: tn = {std::make_shared<Sqrt>(), std::move(node)}; break;
        case EF_TAN: tn = {std::make_shared<Tan>(), std::move(node)}; break;
        case EF_TANH: tn = {std::make_shared<Tanh>(), std::move(node)}; break;
        case EF_TRUNC: tn = {std::make_shared<Trunc>(), std::move(node)}; break;
        default: break;
      }
      node = std::make_unique<TreeNode>(std::move(tn));
    } else if (accept(ET_OPEN)) {
      m_token = nextToken();
      node = list();
      expect(ET_CLOSE);
      m_token = nextToken();
    } else if (accept(ET_TRM)) {
      m_token = nextToken();
      node = factor();
    } else {
      std::cerr << "Error: factor() - syntax error." << std::endl;
      m_token = nextToken();
      node = {};
    }

    if (accept(ET_FAC)) {
      TreeNode tn = {std::make_shared<Fac>(), std::move(node)};
      node = std::make_unique<TreeNode>(std::move(tn));
      m_token = nextToken();
    }
    if (accept(ET_POW)) { // Note: 2^2^3 = 2^(2^3) = 256 (not (2^2)^3 = 64)
      m_token = nextToken();
      UTreeNode right = factor();
      TreeNode tn = {std::make_shared<Pow>(), std::move(node),
                     std::move(right)};
      node = std::make_unique<TreeNode>(std::move(tn));
    }
    optimize(&node);
    return node;
  }

  // <term> : <factor> {('*' | '/' | '%') <factor>}
  UTreeNode term() {
    UTreeNode node = factor();
    while (accept(ET_MUL) || accept(ET_DIV) || accept(ET_MOD)) {
      TreeNode tn;
      auto type = m_token.exprType;
      m_token = nextToken();
      UTreeNode right = factor();
      if (type == ET_MUL)
        tn = {std::make_shared<Mul>(), std::move(node), std::move(right)};
      else if (type == ET_DIV)
        tn = {std::make_shared<Div>(), std::move(node), std::move(right)};
      else tn = {std::make_shared<Mod>(), std::move(node), std::move(right)};
      node = std::make_unique<TreeNode>(std::move(tn));
      optimize(&node);
    }
    return node;
  }

  // <expression> : <term> {('+' | '-') <term>}
  UTreeNode expression() {
    UTreeNode node = term();
    while (accept(ET_ADD) || accept(ET_SUB)) {
      TreeNode tn;
      auto type = m_token.exprType;
      m_token = nextToken();
      UTreeNode right = term();
      if (type == ET_ADD)
        tn = {std::make_shared<Add>(), std::move(node), std::move(right)};
      else tn = {std::make_shared<Sub>(), std::move(node), std::move(right)};
      node = std::make_unique<TreeNode>(std::move(tn));
    }
    optimize(&node);
    return node;
  }

  // <list> : <expression> {',' <expression>}
  UTreeNode list() {
    UTreeNode node = expression();
    for (int arity = 1; accept(ET_SEP); ++arity) {
      if (arity == 1) {
        m_args.clear();
        m_args.push_back(std::move(node));
      }
      m_token = nextToken();
      m_args.push_back(std::move(expression()));
    }
    return node;
  }

  void compile() {
    std::cout << "Compiled: ";
    m_token = nextToken();
    UTreeNode node = expression();
    if (node) m_expressions.push_back(std::move(node));
    std::cout << std::endl;
  }

private:
  std::vector<UTreeNode> m_expressions; // compiled final expressions
  std::vector<UTreeNode> m_args;        // current arguments for a function
  Token m_token;                        // current token
  const char* m_exprStr{nullptr};       // input expression string
};

} // namespace me

#endif // MATH_EXPRESSION_H
