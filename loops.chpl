

config const n = 100;

//
// Local array A, initialized with 1 to n
//
var A: [0..#n] int;
A = 1..#n;


//
// Serial execution
//
for a in A {
  writeln(a);
}

//
//  Parallel execution
//
forall a in A {
  writeln(a);
}


//
// Distributed array B, initialized with 1 to n
//
use BlockDist;
const D = blockDist.createDomain(0..#n);
var B: [D] int;
B = 1..#n;

//
// Distributed execution
//
forall b in B {
  writeln(b);
}
