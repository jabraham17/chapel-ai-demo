

use NDArray;
use Remote;
use Autograd;
import Utilities as util;
use Utilities.Standard;

type tensor = staticTensor(?);

record staticTensor : serializable {
    param rank: int;
    type eltType = real(64);
    var resource: shared BaseTensorResource(eltType,rank);
    forwarding resource only to, array, grad, device ;//backward;
    proc meta do return this.resource;

    proc _dom do return resource.array.domain;

    proc init(param rank: int, type eltType = real(64)) {
        this.rank = rank;
        this.eltType = eltType;
        this.resource = new shared TensorResource(eltType,rank,new baseValue());
    }

    proc init(in resource: shared BaseTensorResource(?eltType,?rank)) {
        this.rank = rank;
        this.eltType = eltType;
        this.resource = resource;
    }

    proc init(in resource: owned BaseTensorResource(?eltType,?rank)) {
        this.init(shared.adopt(resource));
    }

    proc init(array: ndarray(?rank,?eltType)) {
        this.rank = rank;
        this.eltType = eltType;
        on Remote.defaultDevice var ar: shared Remote(ndarray(rank,eltType)) = array;
        this.resource = new shared TensorResource(ar);
    }

    proc init(dom: domain(?),type eltType = real) {
        const normal = util.normalizeDomain(dom);
        param rank = normal.rank;
        on Remote.defaultDevice var ar: shared Remote(ndarray(rank,eltType)) = new ndarray(normal,eltType);
        this.rank = rank;
        this.eltType = eltType;
        this.resource = new shared TensorResource(ar);
    }

    proc init(arr: [] ?eltType) do
        this.init(new ndarray(arr));

    proc init(it: _iteratorRecord) {
        const arr = it;
        this.init(arr);
    }

    proc this(args...) {
        return this.slice((...args));
    }

    proc reshapeDomain(dom: this._dom.type) {
        on this.device {
            ref arr = this.array;
            ref gra = this.grad;
            arr.reshapeDomain(dom);
        }
    }

    proc _setArrayData(value) {
        on this.device {
            const devVal = value;
            ref dat = this.array.data;
            dat = devVal;
        }
    }

    proc detach(copy: bool = true, keepGrad: bool = false): staticTensor(rank,eltType) {
        return new staticTensor(meta.detach(copy,keepGrad));
    }
}

operator :(in t: staticTensor(?rank,?eltType), type toType): staticTensor(rank,toType) {
    const a = t.array;
    const b = a : toType;
    return new staticTensor(b);
}

proc tensorFromCtx(param rank: int, type eltType, ctx): staticTensor(rank,eltType) {
    var newMeta = new owned TensorResource(eltType,rank,ctx);
    newMeta.forward();
    return new staticTensor(newMeta);
}


operator +(a: staticTensor(?rank,?eltType), b: staticTensor(rank,eltType)) {
    var ctx = new addOp(rank,eltType,a.meta,b.meta);
    return tensorFromCtx(rank,eltType,ctx);    
}

operator -(a: staticTensor(?rank,?eltType), b: staticTensor(rank,eltType)) {
    var ctx = new subOp(a.meta,b.meta);
    return tensorFromCtx(rank,eltType,ctx); 
}

operator *(a: staticTensor(?rank,?eltType), b: staticTensor(rank,eltType)) {
    var ctx = new multOp(rank,eltType,a.meta,b.meta);
    return tensorFromCtx(rank,eltType,ctx);
}

operator /(a: staticTensor(?rank,?eltType), b: staticTensor(rank,eltType)) {
    var ctx = new divOp(a.meta,b.meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.reshape(dom: domain(?)) {
    param newRank = dom.rank;
    var ctx = new reshapeOp(rank,newRank,eltType,dom.shape,meta);
    return tensorFromCtx(newRank,eltType,ctx);
}
proc staticTensor.reshape(newShape: int ...?newRank) {
    var ctx = new reshapeOp(rank,newRank,eltType,newShape,meta);
    return tensorFromCtx(newRank,eltType,ctx);
}

proc staticTensor.relu() {
    var ctx = new reluOp(meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.permute(axes: int...rank) {
    var ctx = new permuteOp(rank,eltType,axes,meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.expand(axes: int...rank) {
    var ctx = new expandOp(rank,eltType,axes,meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.pad(args: (2 * int)...rank, value: eltType = 0.0) {
    var ctx = new padOp(rank,eltType,args,value,meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.shrink(args: (2*int)...rank) {
    var ctx = new shrinkOp(rank,eltType,args,meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.slice(dom: domain(?)) where dom.rank == rank {
    var ctx = new sliceOp(rank,eltType,dom,meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.slice(rngs: range...rank) {
    const dom = {(...rngs)};
    var ctx = new sliceOp(rank,eltType,dom,meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.sum(axes: int...?r) {
    if rank - r < 0 {
        compilerError("Cannot sum more axes than rank. ");
    }
    var ctx = new sumOp(rank,eltType,r,axes,meta);

    param newDim = if rank - r == 0 then 1 else rank - r;
    return tensorFromCtx(newDim,eltType,ctx);
}

proc staticTensor.unsqueeze(dim: int): staticTensor(rank + 1,eltType) {
    const shape = this.array.domain.shape;
    param newRank: int = rank + 1;
    var offset: int = 0;
    var newShape: newRank * int;
    for param i in 0..<newRank {
        if i == dim {
            newShape(i) = 1;
            offset = 1;
        } else {
            newShape(i) = shape(i - offset);
        }
    }
    return this.reshape((...newShape));
}

proc staticTensor.max(): staticTensor(1,eltType) {
    var ctx = new maxOp(rank,eltType,rank,this.array.shape,meta);
    return tensorFromCtx(1,eltType,ctx);
}

proc staticTensor.exp(): staticTensor(rank,eltType) {
    var ctx = new expOp(meta);
    return tensorFromCtx(rank,eltType,ctx);
}

proc staticTensor.softmax(): staticTensor(rank,eltType) {

    const myShape = this.array.data.domain.shape;

    var baseShape: rank * int;
    for param i in 0..<rank do
        baseShape(i) = 1;

    var sumAxes: rank * int;
    for param i in 0..<rank do
        sumAxes(i) = i;

    var memx = this.max().reshape((...baseShape)).expand((...myShape));
    var m = this - memx;
    var e = m.exp();
    var ss = e.sum((...sumAxes)).reshape((...baseShape)).expand((...myShape));
    return e / ss;
}


proc matvec(mat: staticTensor(2,?eltType),vec: staticTensor(1,eltType)): staticTensor(1,eltType) {
    const (n,) = vec.array.domain.shape;
    const (m,_n) = mat.array.domain.shape;
    if n != _n then halt("arrays must be same shape" + n : string + " " + _n : string);
    var vec_ = vec.reshape(1,n);
    var v = vec_.expand(m,n);
    var Mv = mat * v;
    return Mv.sum(1);
}

proc matvec(mat: staticTensor(2,?eltType),vec: staticTensor(2,eltType)): staticTensor(2,eltType) {
    const (b,n) = vec.array.domain.shape;
    const (m,_n) = mat.array.domain.shape;
    if n != _n then halt("arrays must be same shape" + n : string + " " + _n : string);
    var vec_ = vec.reshape(b,1,n);
    var v = vec_.expand(b,m,n);
    var M_ = mat.reshape(1,m,n);
    var M = M_.expand(b,m,n);
    var Mv = M * v;
    return Mv.sum(2);
}

proc type staticTensor.matvecmul(m,v) {
    return matvec(m,v);
}

proc type staticTensor.convolve(features: staticTensor(3,?eltType),kernel: staticTensor(4,eltType), stride: int, padding: int): staticTensor(3,eltType) {
    var ctx = new conv2DOp(eltType,features.meta,kernel.meta,stride,padding);
    return tensorFromCtx(3,eltType,ctx);
}

proc type staticTensor.convolve(features: staticTensor(3,?eltType),kernel: staticTensor(4,eltType), bias: staticTensor(1,eltType), stride: int, padding: int): staticTensor(3,eltType) {
    // on here.gpus[0] var x: shared Remote(ndarray(3,eltType)) = ndarray.convolve(features.array,kernel.array,bias.array,stride);

    var t = new staticTensor(3,eltType);
    on t.device {
        t.array = ndarray.convolve(features.array,kernel.array,bias.array,stride, padding);
    }
    return t;
}


proc type staticTensor.matvecmulFast(mat: staticTensor(2,?eltType),vec: staticTensor(1,eltType)): staticTensor(1,eltType) {
    var u = new staticTensor(1,eltType);
    on u.device {
        u.array = ndarray.matvecmul(mat.array,vec.array);
    }
    return u;
}


proc type staticTensor.topk(t: staticTensor(1,?eltType), k: int): staticTensor(1,int) {
    var u = new staticTensor(1,int);
    on u.device {
        u.array = t.array.topk(k);
    }
    return u;
}

proc staticTensor.dilate(dil: int): staticTensor(3,eltType) where this.rank == 3 {
    var dilated = new staticTensor(3,eltType);
    on this.device {
        ref dat = this.array;
        ref dila = dilated.array;
        const d = dat.dilate(dil);
        dila.reshapeDomain(d.domain);
        dila = d;
    }
    return dilated;
}

proc staticTensor.maxPool(poolSize:int) do return this.maxPool(poolSize,poolSize);
proc staticTensor.maxPool(poolSize: int, stride: int): staticTensor(3,eltType) where this.rank == 3 {
    var pool = new staticTensor(3,eltType);
    on this.device {
        ref dat = this.array;
        ref pl = pool.array;
        const p = ndarray.maxPool(dat,poolSize, stride);
        pl.reshapeDomain(p.domain);
        pl = p;
    }
    return pool;
}

// adaptiveAvgPool2d
proc staticTensor.adaptiveAvgPool2d(outputSize: int): staticTensor(3,eltType) where this.rank == 3 {
    var pool = new staticTensor(3,eltType);
    on this.device {
        ref dat = this.array;
        ref pl = pool.array;
        const p = ndarray.adaptiveAvgPool2d(dat,outputSize);
        pl.reshapeDomain(p.domain);
        pl = p;
    }
    return pool;
}

proc type staticTensor.arange(to: int,type eltType = real,shape: ?rank*int): staticTensor(rank,eltType) {
    const dom = util.domainFromShape((...shape));
    const A: [dom] eltType = foreach (_,x) in zip(dom,0..<to) do x:eltType;
    return new staticTensor(A);
}

proc type staticTensor.arange(shape: int...?rank): staticTensor(rank,real) {
    const _shape: rank * int = shape;
    const dom = util.domainFromShape((..._shape));
    const to = dom.size;
    const A: [dom] real = foreach (_,x) in zip(dom,0..<to) do x:real;
    return new staticTensor(A);
}


proc type staticTensor.fromShape(type eltType = real,shape: int...?rank,value: eltType = (0:eltType)): staticTensor(rank,eltType) {
    const v = value;
    const dom = util.domainFromShape((...shape));
    const A: [dom] eltType;
    var t = new staticTensor(A);
    return t;
}

proc type staticTensor.zeros(shape: int...?rank): staticTensor(rank,real) do
    return staticTensor.fromShape(real,(...shape),0.0);

proc type staticTensor.zeros(type eltType,shape: int...?rank): staticTensor(rank,eltType) do
    return staticTensor.fromShape(eltType,(...shape),0 : eltType);

proc type staticTensor.ones(shape: int...?rank): staticTensor(rank,real) do
    return staticTensor.fromShape(real,(...shape),value=1.0);

proc type staticTensor.ones(type eltType,shape: int...?rank): staticTensor(rank,eltType) do
    return staticTensor.fromShape(eltType,(...shape),value=1 : eltType);



config const n = 100;
config const diag = false;
config const size = 3;

proc main() {



    if diag {
        use GpuDiagnostics;

        startGpuDiagnostics();
        startVerboseGpu();
    }

    // arange(15,real,(3,5));

    var t = new staticTensor(2,real);
    t.array.reshapeDomain({0..<3,0..<5});
    t.to(Remote.defaultDevice);
    on t.device {
        ref tarr = t.array;
        ref tdata = tarr.data;
        // tdata += 1.0;
        // foreach i in tdata.domain do
        //     tdata[i] = tdata[i] + 1.0;
        // tdata = foreach x in tdata do x + 1.0; // causes grained kernel launches 
        // @assertOnGpu
        forall i in tarr.data.domain.every() do
            tdata[i] = tarr.data[i] + 1.0;
    }


    const run1 = false;
    if run1 {
        var M = staticTensor.arange(15,real,(5,3));
        writeln(M);
        var u = staticTensor.arange(3,real,(1,3));
        writeln(u);

        var x = u.expand(5,3);
        writeln(x);

        var Mx = M * x;
        writeln(Mx);

        var y = Mx.sum(1);
        writeln(y);


        var u_ = staticTensor.arange(3,real,(3,));
        var y_ = matvec(M,u_);

        writeln(y_);
        var z = y_.sum(0);
        writeln(z);

        // z.backward();


        writeln(M.grad);
    }



    var M = staticTensor.arange(15,real,(5,3));
    writeln(M);

    var x = staticTensor.arange(9,real,(3,3));
    writeln(x);

    var y = matvec(M,x);
    writeln(y);

    // y.sum(0).sum(0).backward();
    // writeln(M.grad);



    const run2 = false;
    if run2 {
        var W = M.grad;
        var Q = W.shrink((1,3),(1,2));
        writeln(Q);
        writeln(Q.domain);

        var U = W.pad((0,3),(0,0));
        writeln(U);
    }

    const run3 = false;
    if run3 {
        var W = staticTensor.ones(5,3);
        var Q = W.shrink((1,3),(1,2));
        writeln(Q);

        var U = W.pad((1,3),(0,0),68);
        writeln(U);

        U.slice(0..2,0..2).sum(0).sum(0).backward();
        U[0..2,0..2].sum(0).sum(0).backward();

        writeln(W.grad);

        writeln(staticTensor.arange(5,2));

        var a = staticTensor.arange(4);
        writeln(a);
        writeln(a.unsqueeze(1));


        var img = staticTensor.arange(3,9,9);
        var ker = staticTensor.arange(1,3,3,3);
        var fet = staticTensor.convolve(img,ker,2,0);
        writeln(fet);

        var b = staticTensor.arange(1,3,3);

        writeln(b.dilate(1));
        writeln(b.dilate(1).maxPool(2));
    }




    var img: staticTensor(3,real) = staticTensor.arange(1,9,9);
    writeln(img);

    var ker = staticTensor.arange(1,1,3,3);
    var fet = staticTensor.convolve(img,ker,1,0);
    writeln("Features:", fet);
    var sm = fet.sum(0).sum(0).sum(0);
    writeln(sm);
    sm.backward();
    writeln(img.grad);
    writeln(ker.grad);


    // {
    //     writeln("Begin");
    //     var x = staticTensor.arange(3,5);
    //     writeln(x);
    //     // writeln(x.array);
    //     on x.device { x.array = x.array.reshape(5,4);}
    //     // writeln(x.array.shape);
    //     writeln(x);
    //     on x.device { x.array = x.array.reshape(1,5,4).reshape(3,5);}
    //     writeln(x);

    // }

    // {
    //     var x = staticTensor.arange(10);
    //     writeln(x);
    //     var y = x.sum(0);
    //     writeln(y);
    //     y.backward();
    //     writeln(x.grad);
    //     writeln(y.grad);
    // }

    // inline iter _domain.each {
    //     for i in 0..<this.size {
    //         yield this.orderToIndex(i);
    //     }
    // }

    // const R = 0..<10;
    // writeln(R,R.type:string);

    // const D = {R,R};
    // writeln(D,D.type:string);

    // const D2: util.Types.stdDomain = {R,R};
    // writeln(D2,D2.type:string);

    // const D = {0..<3,0..<5};
    // foreach (a,b) in D.each do
    //     writeln((a,b));

    // img = staticTensor.arange(1,9,9);
    // ker = staticTensor.arange(1,1,3,3);
    // fet = staticTensor.convolve(img,ker,2,0);
    // sm = fet.sum(0).sum(0).sum(0);
    // writeln(sm);
    // sm.backward();
    // writeln(fet.array.shape);
    // writeln(fet);
    // writeln(img.grad);
    // writeln(ker.grad);
    // foreach i in img.array.domain with (ref img) {
    //     img.array.data[i] = 2.0;
    // }

    // writeln(x.array.data[1,0]);

    // const ar = arange(15,real,(3,5));
    // var t = new staticTensor(ar);
    // t.to(here.gpus[0]);
    // // writeln(ar.data.locale);
    // // writeln(t.array.data.locale);
    // on t.device {
    //     ref tarr = t.array;
    //     ref tData = tarr.data;
    //     var res = t.meta.dataResource;
    // }

    // var at = new staticTensor(arange(15,real,(3,5)));
    // var bt = new staticTensor(arange(15,real,(3,5)));
    // // writeln(a.array.data.locale,b.array.data.locale);
    // const ar: ndarray(2,real) = arange(15,real,(3,5));
    // var a = new remote(ar);
    // var b = new remote(ar);
    // writeln(a.access().data.locale,b.access().data.locale);

    // var c = a + b;
    // writeln(a.access().data.locale,b.access().data.locale);
    // var ct = at + bt;

    // var arr1 = new ndarray({0..size,0..size,0..size});
    // var arr2 = new ndarray({0..size,0..size,0..size});

    // var t1 = new staticTensor(arr1);
    // var t2 = new staticTensor(arr2);

    // var t1 = new staticTensor(3,real);
    // var t2 = new staticTensor(3,real);
    // t1.array.reshapeDomain({0..size,0..size,0..size});
    // t2.array.reshapeDomain({0..size,0..size,0..size});
    // var t3 = t1 + t2;
    // writeln(t3.array);

    // var t4 = t3.sum(0,1);
    // writeln(t4.array);

    // writeln("-----------------------------");

    // var t = new staticTensor(2,real);
    // t.array.reshapeDomain({0..<3,0..<5});
    // for (i,n) in zip(t.array.domain,0..<15) do
    //     t.array.data[i] = n;
    // writeln(t.array.data,"\n -------------- ");

    // var u = t.sum(0);
    // writeln(u.array);

    // var w = u.sum(0);
    // writeln(w.array);

    // var x = t.sum(1).sum(0);
    // writeln(x.array);

    // var y = (t + t).sum(0,1);
    // writeln(y);
    // writeln(y);


    // writeln(t.grad);

    // y.resource.backward();

    // writeln(t.grad);

    // y.resource.backward();
    // writeln(t);

    // var z = arange(15,real,(3,5));
    // writeln(z);

    // var T = new staticTensor(z);

    // var s = (T * T).sum(0,1);
    // writeln(s);
    // s.resource.backward();
    // writeln(T.grad);

    // var X = X.expand();
    // for i in 0..n {
    //     t3 = t3 + t1 + t2;
    // }



    // var input1 = new shared TensorResource(arr1,new baseValue());
    // var input2 = new shared TensorResource(arr2,new baseValue());
    // var sum = new shared TensorResource(1,real(64), new addOp(1,real,input1,input2));

    // var t1 = new staticTensor(input1);
    // var t2 = new staticTensor(input2);
    // var t3 = new staticTensor(sum);

    // writeln(t1);
    // writeln(t2);
    // writeln(t3.array);

    // t3.forward();
    // writeln(t3.array);

    // writeln(t3.type:string);

    // var t4 = t1 + t2;
    // writeln(t1.data);

    // // writeln((t1 * t2).data);

    // var x = (t1 * t2).reshape({0..1});


    // writeln(x.array);

    // var rl = (t2 * t1).relu();
    // writeln(rl.array);

    // var matInput = for (i,j) in {0..<2,0..<3} do arr1[i] * arr2[j];

    // var mat = new staticTensor(new ndarray(matInput));
    // writeln(mat.array.shape,mat.array);

    // var prm = mat.permute(1,0);
    // writeln(prm.array.shape,prm.array);

    // writeln((t4.meta : shared TensorResource(1,real,addOp(1,real))).operationData.backward(t4.array));

    // var mInput = for (i,j) in {0..<3,0..<1} do i * 10.0 + j + 1;
    // var m = new ndarray(mInput);
    // writeln(m.data,m.shape);
    // var mExpanded = m.expand(3,4);
    // writeln(mExpanded.data,mExpanded.shape);

}





// proc staticTensor.serialize(writer: IO.fileWriter(locking=false,?sr1),ref serializer: ?sr2) {
//     serializer.beginRecord()
// }

import IO;
// pretty printing
proc staticTensor.serialize(writer: IO.fileWriter(locking=false, IO.defaultSerializer),ref serializer: IO.defaultSerializer) {
    // const name = "ndarray(" + rank:string + "," + eltType:string + ")";
    // var ser = serializer.startRecord(writer,name,2);
    // ser.writeField("shape",this.data.shape);
    // // var serArr = ser.startArray();
    // ser.writeField("data",this.data);
    // ser.endRecord();

    const prevDev = this.device;
    this.to(here);


    writer.write("tensor(");
    const shape = this.array.shape;
    var first: bool = true;
    for (x,i) in zip(this.array.data,0..) {
        const idx = util.nbase(shape,i);
        if idx[rank - 1] == 0 {
            if !first {
                writer.write("\n       ");
            }
            writer.write("[");
        }
        writer.writef("%{##.#}",x);
        
        if idx[rank - 1] < shape[rank - 1] - 1 {
            if rank == 1 then
                writer.write("  ");
            else
                writer.write("  ");
        } else {
            writer.write("]");
        }
        first = false;
    }
    writer.write(",\n       shape = ",this.array.data.shape);
    writer.write(",\n       rank = ",this.rank);
    writer.writeln(")");

    this.to(prevDev);
}


// chapel generic one
proc staticTensor.serialize(writer: IO.fileWriter(?),ref serializer: ?srt2) where srt2 != IO.defaultSerializer {

    const prevDev = this.device;
    this.to(here);

    var rh = serializer.startRecord(writer,"tensor",3);
    rh.writeField("rank",rank);
    rh.writeField("eltType",eltType:string);
    rh.writeField("resource",resource);
    rh.endRecord();

    this.to(prevDev);
}

proc staticTensor.serialize(writer: IO.fileWriter(locking=false, IO.defaultSerializer),ref serializer: IO.defaultSerializer,param capitalT: bool) where capitalT == true {
    const prevDev = this.device;
    this.to(here);

    writer.write("Tensor(");
    const shape = this.array.data.domain.shape;
    var first: bool = true;
    for (x,i) in zip(this.array.data,0..) {
        const idx = util.nbase(shape,i);
        if idx[rank - 1] == 0 {
            if !first {
                writer.write("\n       ");
            }
            writer.write("[");
        }
        writer.writef("%{##.#}",x);
        
        if idx[rank - 1] < shape[rank - 1] - 1 {
            if rank == 1 then
                writer.write("  ");
            else
                writer.write("  ");
        } else {
            writer.write("]");
        }
        first = false;
    }
    writer.write(",\n       shape = ",this.array.data.shape);
    writer.write(",\n       rank = ",this.rank);
    writer.writeln(")");

    this.to(prevDev);
}

proc ref staticTensor.read(fr: IO.fileReader(?)) throws {
    var arr = this.array;
    arr.read(fr);
    on this.device {
        const devArr = arr;
        ref ar = this.array;
        ar = devArr;
    }
}
