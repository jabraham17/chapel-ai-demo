module Remote {


class _RemoteVarContainer : serializable {
    var containedValue;
} // meow


class Remote : serializable {
    type eltType;
    var value: owned _RemoteVarContainer(eltType);

    proc ref get() ref do
        return this;

    proc ptr ref do 
        return value.containedValue;

    proc device: locale do
        return ptr.locale;


    proc init(in value: owned _RemoteVarContainer(?eltType)) {
        this.eltType = eltType;
        this.value = value;
    }

    proc init(value: ?eltType, device: locale = Remote.defaultDevice) where !isSubtype(eltType,Remote(?)) {
        compilerWarning("This initializer doesn't offer the best performance. ");
        var v: owned _RemoteVarContainer(eltType)?;
        on device do v = new _RemoteVarContainer(value);
        this.init(try! v : owned _RemoteVarContainer(eltType));
    }

    proc init(type eltType, in tr: _thunkRecord, device: locale = Remote.defaultDevice) {
        var v: owned _RemoteVarContainer(eltType)?;
        on device {
            var val: eltType = tr;
            v = new _RemoteVarContainer(val);
        }
        this.init(try! v : owned _RemoteVarContainer(eltType));
    }

    proc init(type eltType, device: locale = Remote.defaultDevice) {
        var v: owned _RemoteVarContainer(eltType)?;
        on device {
            var val: eltType;
            v = new _RemoteVarContainer(val);
        }
        this.init(try! v : owned _RemoteVarContainer(eltType));
    }

    proc copyContainer(dest: locale = device): owned _RemoteVarContainer(eltType) {
        var c: owned _RemoteVarContainer(eltType)?;
        on dest {
            var val: eltType = ptr;
            c = new _RemoteVarContainer(val);
        }
        return try! c : owned _RemoteVarContainer(eltType);
    }

    proc copyTo(dest: locale = device): owned Remote(eltType) do
        return new Remote(copyContainer(dest));

    proc copy(): owned Remote(eltType) do
        return copyTo(device);

    proc to(dest: locale) {
        if dest == device then return;
        value = copyContainer(dest);
    }
}

proc ref (owned Remote(?eltType)).get() ref : owned Remote(eltType) do
    return this;

proc ref (shared Remote(?eltType)).get() ref : shared Remote(eltType) do
    return this;

proc ref (unmanaged Remote(?eltType)).get() ref : unmanaged Remote(eltType) do
    return this;

proc type Remote.defaultDevice: locale do
    return if here.gpus.size >= 1 then here.gpus[0] else here;


public use ChapelRemoteVars;

private inline proc __defaultValue(type t) {
    var tmp: t;
    return tmp;
}

inline proc chpl__buildRemoteWrapper(loc: locale, type inType: Remote(?eltType)) do
    return chpl__buildRemoteWrapper(loc, inType, __primitive("create thunk", __defaultValue(eltType)));

inline proc chpl__buildRemoteWrapper(loc: locale, type inType: owned Remote(?eltType), in tr: _thunkRecord)  {
    var c: owned _RemoteVarContainer(eltType)?;
    on loc {
        var forced: eltType = __primitive("force thunk", tr);
        c = new _RemoteVarContainer(forced);
    }
    return new owned Remote(try! c : owned _RemoteVarContainer(eltType));
}
inline proc chpl__buildRemoteWrapper(loc: locale, type inType: shared Remote(?eltType), in tr: _thunkRecord)  {
    var c: owned _RemoteVarContainer(eltType)?;
    on loc {
        var forced: eltType = __primitive("force thunk", tr);
        c = new _RemoteVarContainer(forced);
    }
    return new shared Remote(try! c : owned _RemoteVarContainer(eltType));
}
}

