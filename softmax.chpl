use GpuDiagnostics;
import RangeChunk;
use Random;
use Math;

/* Config constants automatically get command line flags */
config const n = 100;
config const print = false;
config const verboseGpu = false;

/* Determine the compute nodes to use.
   For the CPU case we are using an array of the same locale. */
const hasGpus = here.gpus.size > 0;
const nLocales = if hasGpus then here.gpus.size else here.maxTaskPar;
var targetLocales: [0..<nLocales] locale;
if hasGpus then targetLocales = here.gpus;
           else targetLocales = here;

/* Create a local array that will be chunked up across the GPUs (or CPU tasks) */
const elmRange = 0..<n;
var HostArr: [elmRange] real(32);
const nChunks = nLocales;

/* Initialize the array to get more interesting data */
fillRandom(HostArr);

if verboseGpu then startVerboseGpu();

/* Softmax is defined as `Xi = (exp(Xi) / sum(exp(X)) )` */

/* Applies the exponential function to each element and sums it */
var expSum: real(32);
if hasGpus {
  /* The `+ reduce expSum` tells the compiler our intention */
  coforall i in 0..<nChunks with (+ reduce expSum) {
    on targetLocales[i] {
      const chunk = RangeChunk.chunk(elmRange, nChunks, i);
      var DeviceArr = HostArr[chunk];
      /* This is a kernel launch to compute the sum on each GPU */
      expSum += + reduce exp(DeviceArr);
    }
  }
} else {
  /* For the CPU case (or for a single GPU), we can just write one expression */
  expSum = + reduce exp(HostArr);
}

/* Using the sum we just computed, we can finish the softmax function */
coforall i in 0..<nChunks {
  on targetLocales[i] {
    const chunk = RangeChunk.chunk(elmRange, nChunks, i);
    var DeviceArr = HostArr[chunk];
    /* This expression will also result in a kernel launch */
    DeviceArr = exp(DeviceArr) / expSum;
    HostArr[chunk] = DeviceArr;
  }
}

if verboseGpu then stopVerboseGpu();

if print then writeln(HostArr);
