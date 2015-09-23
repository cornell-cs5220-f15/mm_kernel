# Matrix multiply kernel tuning

N.B.  This document (code harness, notes, etc) are from the S14
offering of CS 5220.  The totient cluster has a newer architecture
with wider vector units, so it's likely possible to do better than
this.

There are fundamentally two ways to lose performance for matrix
multiplication, mirroring the two points mentioned in lecture 3:

1. We can lose performance because of poor cache utilization.
2. We can lose performance because of poor processor utilization.

The performance loss due to memory access times may be the more
counterintuitive of these two issues, but it is also the one that is
perhaps most straightforward to deal with by techniques such as
blocking and copy optimization for improved layout and alignment.  The
performance loss due to poor processor utilization, though, is a bit
more counterintuitive.

## Building on fast kernels

It frequently does not make sense to expend the effort required to get
the best possible performance.  Pretty good performance may be good enough,
and there's often a law of diminishing returns -- once you've eliminated the
easy bottlenecks, the only opportunities that remain require a great deal
of work.  If we're going to make our code a mess in order to squeeze out as
much performance as possible, it makes good sense to at least try to contain
the mess.  That suggests that if it's necessary to write black magic code,
we should at the very least try to tuck it into a small kernel so that
a later reader does not have to exhaust himself.

If we're going to start from a fast kernel, it also makes sense to
give ourselves the instruments to see whether we're succeeding in
creating something fast from which we can build higher-level codes.
That means we need a timer.  The `ktimer.c` module is exactly such a
timer.  We assume a fixed-size kernel matrix multiply (`kdgemm`) that
takes in pre-digested data that is aligned and laid out however we'd
like.  The kernel routine has to tell us the size of the matrices it
will accept, and from there we run a timing loop to compute an
effective gigaflop rate.

It's also useful to automatically test the kernel in this harness.

## Principles for a fast kernel

There are a few principles involved in getting good instruction-level
parallelism, and then there is some black magic.  Some principles are:

1. Ask for the compiler's help!  At the very least, one should use an
   appropriate set of base optimizations (`-O2` or `-O3`) and tell the
   compiler what type of instruction set it should target for tuning.

2. Organize for independent subcomputations.  For example, if we
   wanted to take the sum of four numbers, the expression
   `(a+b)+(c+d)` involves two adds that can be scheduled in parallel
   by the processor, while in the expression `((a+b)+c)+d`, though
   mathematically equivalent, the processor is forced to serialize the
   additions.

3. Provide the compiler with a good mix of instructions.  On the
   Nehalem, the best we can do is to retire one vector add and one
   vector multiply at each cycle.  That means the compiler needs to
   have a mix of adds and multiplies visible when it is doing
   instruction scheduling.

In an ideal world, a vectorizing compiler would be smart enough to
carry things from here.  We don't live in an ideal world, though, an
while the Intel compiler optimizer does some remarkable things (and
the GNU compiler optimizers have gotten better, too), they still can
miss some things.  Using the SSE or AVX primitives directly can boost
performance significantly over the default vectorization.  The "right"
way to do this, though, is still to get mileage out of the things the
compiler is good at (i.e. instruction scheduling).  That means that we
aren't going to drop all the way down to assembler.  Instead, we'll
use the intrinsics provided by the compiler.

## Black magic for a fast kernel

The Nehalem architecture can launch four instructions simultaneously,
assuming they are the right types.  Floating point SSE add and
multiply are on separate ports, so one of each can be launched in a
single cycle.  But in addition to adds, multiplies, loads, and stores,
it is also helpful to do some data shuffling operations.  There are
multiple instructions for shuffling data, and they use different
ports.  If we view the 128-bit register as two 64-bit doubles, the
most obvious instruction (`shufpd`) uses the same port as the floating
point multiply.  However, we can achieve the same goal by treating the
128-bit register as four 32-bit chunks and permuting those chunks
appropriately, and that uses a different port.  This makes no
difference whatsoever until the performance gets above a critical
threshold, but it starts to make a significant difference once we've
got a fast enough kernel.

Apart from the abuse of the shuffle instructions, the other bit of
"black magic" in this kernel goes into the layout of the matrix
multiply itself.  At its core, this routine works by computing 2-by-2
outer products with SSE instructions.  The main observation is that we
can compute such a product with a vector multiply to get the diagonal
elements, a shuffle, and a second vector multiply go get the
off-diagonals, based on the formulas:

    c11 = a1*b1,    c12 = a1*b2
    c22 = a2*b2,    c21 = a2*b1

To compute a 2-by-P by P-by-2 matrix vector product, we sum the results
of several such outer products.  For example, to compute a 2-by-2-by-2
multiply, we have the following

     // First outer product
     __m128d a0 = _mm_load_pd(A+2*k);
     __m128d b0 = _mm_load_pd(B+2*k);
     __m128d td0 = _mm_mul_pd(a0, b0);
     __m128d bs0 = swap_sse_doubles(b0);
     __m128d to0 = _mm_mul_pd(a0, bs0);

     // Second outer product
     __m128d a1 = _mm_load_pd(A+2*k+2);
     __m128d b1 = _mm_load_pd(B+2*k+2);
     __m128d td1 = _mm_mul_pd(a1, b1);
     __m128d bs1 = swap_sse_doubles(b1);
     __m128d to1 = _mm_mul_pd(a1, bs1);

     // Sum the two outer products
     __m128d td_sum = _mm_add_pd(td0, td1);
     __m128d to_sum = _mm_add_pd(to0, to1);

     // Load old diagonal and off-diagonals from storage
     __m128d cd = _mm_load_pd(C+0);
     __m128d co = _mm_load_pd(C+2);
     
     // Accumulate 2-by-2-by-2 result
     cd = _mm_add_pd(cd, td_sum);
     co = _mm_add_pd(co, to_sum);
     
     // Write back
     _mm_store_pd(C+0, cd);
     _mm_store_pd(C+2, co);

It would take two more shuffles to get each 2-by-2 product matrix into
a standard row-major or column major order, but there's no real point
in doing things that way.  Instead, we'll just keep the diagonal elements
together and the off-diagonal elements together until we're forced to
switch things up based on the programmer interface.

