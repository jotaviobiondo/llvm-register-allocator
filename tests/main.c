#include <stdio.h>

void print(double d);
double calc(double a, double b, double c);

int main() {
    /*int a, b, c,d=7, f=49, g=29;
    a = 27 + g;
    b = 23 + f;
    a = a + b;
    c = b + 17;*/
    
  int a = 1;
  int b = 3;
  double c = 5.7;
  double x = calc(a, b, c);
  print(c);
  print(x);

  return 0;
}

void print(double d) {
  printf("%g\n", d);
}

double calc(double a, double b, double c) {
  return c * b * b * a + 150 * b + c;
}
