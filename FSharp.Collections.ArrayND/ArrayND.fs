namespace FSharp.Collections

open GenericTensor.Core
module ArrayNDHelpers =
    exception BroadcastException of longShape: TensorShape * shortShape: TensorShape with
        override this.Message = $"Shapes {this.longShape} and {this.shortShape} cannot be broadcast to a single shape."
    exception EmptyDimensionException of shape: TensorShape * dimension: int * operation: string with
        override this.Message = $"Dimension {this.dimension} of shape {this.shape} cannot be reduced because it is empty and the reduction operation {this.operation} has no identity value."
    type [<Struct>] ArrayNDWrapper<'T>(_unused: int) =
        static member val internal Add: 'T -> 'T -> 'T = Unchecked.defaultof<_> with get, set
        static member val internal Subtract: 'T -> 'T -> 'T = Unchecked.defaultof<_> with get, set
        static member val internal Multiply: 'T -> 'T -> 'T = Unchecked.defaultof<_> with get, set
        static member val internal Negate: 'T -> 'T = Unchecked.defaultof<_> with get, set
        static member val internal Divide: 'T -> 'T -> 'T = Unchecked.defaultof<_> with get, set
        static member val internal One: 'T = Unchecked.defaultof<_> with get, set
        static member val internal Zero: 'T = Unchecked.defaultof<_> with get, set
        static member val internal AreEqual: 'T -> 'T -> bool = Unchecked.defaultof<_> with get, set
        static member val internal Serialize: 'T -> byte[] = Unchecked.defaultof<_> with get, set
        static member val internal Deserialize: byte[] -> 'T = Unchecked.defaultof<_> with get, set
        interface IOperations<'T> with
            override _.Add(a, b) = ArrayNDWrapper.Add a b
            override _.Subtract(a, b) = ArrayNDWrapper.Subtract a b
            override _.Multiply(a, b) = ArrayNDWrapper.Multiply a b
            override _.Negate a = ArrayNDWrapper.Negate a
            override _.Divide(a, b) = ArrayNDWrapper.Divide a b
            override _.CreateOne() = ArrayNDWrapper.One
            override _.CreateZero() = ArrayNDWrapper.Zero
            override _.Copy a = a
            override _.AreEqual(a, b) = ArrayNDWrapper.AreEqual a b
            override _.IsZero a = ArrayNDWrapper.AreEqual a ArrayNDWrapper.Zero
            override _.ToString a = string a
            override _.Serialize a = ArrayNDWrapper.Serialize a
            override _.Deserialize a = ArrayNDWrapper.Deserialize a
    type Tensor<'T> = GenTensor<'T, ArrayNDWrapper<'T>>
    let canBroadcast (a: TensorShape) (b: TensorShape) =
        let aLonger = a.Length < b.Length
        let longShape, shortShape = if aLonger then b, a else a, b
        let shortfall = longShape.Length - shortShape.Length
        let mutable ret = true
        for i in 0 .. shortShape.Length - 1 do
            if longShape[i + shortfall] = shortShape[i]
               || longShape[i + shortfall] = 1 || shortShape[i] = 1
            then ()
            else ret <- false
        ret
    let broadcast (a: TensorShape) (b: TensorShape) =
        let aLonger = a.Length < b.Length
        let longShape, shortShape = if aLonger then b, a else a, b
        let shortfall = longShape.Length - shortShape.Length
        TensorShape(longShape.shape |> Array.mapi (fun i long ->
            if i < shortfall then long
            elif long = shortShape[i - shortfall] then long
            elif long = 1 then shortShape[i - shortfall]
            elif shortShape[i - shortfall] = 1 then long
            else raise <| BroadcastException(longShape, shortShape)
        ))
    let getBroadcastedItem (index: int[]) (a: Tensor<'T>) =
        let index = index[index.Length - a.Shape.Length..]
        for i in 0 .. index.Length - 1 do if a.Shape[i] = 1 then index[i] <- 0
        a.GetValueNoCheck index
    let initBeforeIncrement (shape: TensorShape) =
        let ret = Array.zeroCreate<int> shape.Length
        ret[ret.Length - 1] <- -1
        ret
    let increment (index: int[]) (shape: TensorShape) =
        index[index.Length - 1] <- index[index.Length - 1] + 1
        for i in index.Length - 1 .. -1 .. 1 do
            if index[i] >= shape[i] then
                index[i - 1] <- index[i - 1] + 1
                index[i] <- 0
        index[0] < shape[0]
    let initBeforeDecrement (shape: TensorShape) =
        Array.init shape.Length (fun i -> if i = shape.Length - 1 then shape[i] else shape[i] - 1)
    let decrement (index: int[]) (shape: TensorShape) =
        index[index.Length - 1] <- index[index.Length - 1] - 1
        for i in index.Length - 1 .. -1 .. 1 do
            if index[i] < 0 then
                index[i - 1] <- index[i - 1] - 1
                index[i] <- shape[i]
        index[0] >= 0
    let inline operateAlongDimension (dimension: int) (tensor: 'T Tensor) ([<InlineIfLambda>] f: 'T Tensor -> 'U): 'U =
        for i in dimension .. -1 .. 1 do tensor.Transpose(i, i-1)
        let ret = f tensor
        for i in 1 .. dimension do tensor.Transpose(i, i-1)
        ret
    let dimensionReduce (operation: string) (earlyReturn: 'U -> bool) (reduction: 'U -> 'T -> 'U) (dimension: int) (initMap: 'T -> 'U) (finalMap: 'U -> 'V) (tensor: 'T Tensor): 'V Tensor =
        operateAlongDimension dimension tensor (fun tensor ->
            Tensor.CreateTensor(Array.tail tensor.Shape.shape |> TensorShape, fun j ->
                let buffer = Array.zeroCreate tensor.Shape.Length
                let max = tensor.Shape[0] - 1
                if max < 0 then raise <| EmptyDimensionException(tensor.Shape, dimension, operation)
                else
                buffer[0] <- 0
                j.CopyTo(buffer, 1)
                let mutable state = initMap <| tensor.GetValueNoCheck buffer
                let rec loop i =
                    buffer[0] <- i
                    state <- reduction state (tensor.GetValueNoCheck buffer)
                    if earlyReturn state || i = max then state else loop (i + 1)
                if max = 0 then state else loop 1
                |> finalMap
            )
        )
    let dimensionFold (earlyReturn: 'U -> bool) (folder: 'U -> 'T -> 'U) (state: 'U) (dimension: int) (tensor: 'T Tensor) =
        operateAlongDimension dimension tensor (fun tensor ->
            Tensor.CreateTensor(Array.tail tensor.Shape.shape |> TensorShape, fun j ->
                let buffer = Array.zeroCreate tensor.Shape.Length
                j.CopyTo(buffer, 1)
                let mutable state = state
                let max = tensor.Shape[0] - 1
                let rec loop i =
                    buffer[0] <- i
                    state <- folder state (tensor.GetValueNoCheck buffer)
                    if earlyReturn state || i = max then state else loop (i + 1)
                loop 0
            )
        )
open ArrayNDHelpers
type ArrayND<'T> = { Tensor: Tensor<'T> } with
    override this.ToString() = this.Tensor.ToString()
    interface System.Collections.IEnumerable with member this.GetEnumerator() = (this.Tensor.Iterate() |> Seq.map (fun struct(_, x) -> x)).GetEnumerator()
    interface seq<'T> with member this.GetEnumerator() = (this.Tensor.Iterate() |> Seq.map (fun struct(_, x) -> x)).GetEnumerator()
type MaskedArrayND<'T> = ArrayND<'T voption>
module ArrayND =
    // Functions that interface with GenericTensor
    let inline average (arraynd: 'T ArrayND): 'T = Seq.average arraynd
    let inline averaged (dimension: int) (arraynd: 'T ArrayND): 'T ArrayND =
        { Tensor = dimensionReduce "average" (fun _ -> false) (fun struct(sum, count) x -> struct(sum + x, count + 1)) dimension (fun x -> struct(x, 1)) (fun struct(sum, count) -> LanguagePrimitives.DivideByInt sum count) arraynd.Tensor }
    let inline averageBy (projection: 'T -> 'U) (arraynd: 'T ArrayND): 'U = Seq.averageBy projection arraynd
    let inline averageByd (dimension: int) (projection: 'T -> 'U) (arraynd: 'T ArrayND): 'U ArrayND =
        { Tensor = dimensionReduce "averageBy" (fun _ -> false) (fun struct(sum, count) x -> struct(sum + projection x, count + 1)) dimension (fun x -> struct(projection x, 1)) (fun struct(sum, count) -> LanguagePrimitives.DivideByInt sum count) arraynd.Tensor }
    let create (shape: TensorShape) (value: 'T): 'T ArrayND = { Tensor = Tensor.CreateTensor(shape, fun _ -> value) }
    let contains (value: 'T) (arraynd: 'T ArrayND): bool = Seq.contains value arraynd
    let containsd (dimension: int) (value: 'T) (arraynd: 'T ArrayND): bool ArrayND =
        { Tensor = dimensionFold id (fun _ curr -> curr = value) false dimension arraynd.Tensor }
    let countBy (projection: 'T -> 'Key) (arraynd: 'T ArrayND) = Seq.countBy projection arraynd
    let empty<'T> : 'T ArrayND = { Tensor = Tensor<'T>(TensorShape [|0|]) }
    let exists (predicate: 'T -> bool) (arraynd: 'T ArrayND): bool = Seq.exists predicate arraynd
    let exists2 (predicate: 'T -> 'U -> bool) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): bool =
        let shape = broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape
        let rec loop index =
            increment index shape
            && (
            predicate (getBroadcastedItem index arraynd1.Tensor) (getBroadcastedItem index arraynd2.Tensor)
            || loop index)
        loop (initBeforeIncrement shape)
    let forall (predicate: 'T -> bool) (arraynd: 'T ArrayND): bool = Seq.forall predicate arraynd
    let forall2 (predicate: 'T -> 'U -> bool) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): bool =
        let shape = broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape
        let rec loop index =
            not (increment index shape)
            ||
            predicate (getBroadcastedItem index arraynd1.Tensor) (getBroadcastedItem index arraynd2.Tensor)
            && loop index
        loop (initBeforeIncrement shape)
    let find (predicate: 'T -> bool) (arraynd: 'T ArrayND): 'T = Seq.find predicate arraynd
    let findBack (predicate: 'T -> bool) (arraynd: 'T ArrayND): 'T = Seq.findBack predicate arraynd
    let fold (folder: 'State -> 'T -> 'State) (state: 'State) (arraynd: 'T ArrayND): 'State = Seq.fold folder state arraynd
    let fold2 (folder: 'State -> 'T -> 'U -> 'State) (state: 'State) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): 'State =
        let shape = broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape
        let index = initBeforeIncrement shape
        let mutable state = state
        while increment index shape do
            state <- folder state (getBroadcastedItem index arraynd1.Tensor) (getBroadcastedItem index arraynd2.Tensor)
        state
    let foldBack (folder: 'T -> 'State -> 'State) (arraynd: 'T ArrayND) (state: 'State): 'State = Seq.foldBack folder arraynd state
    let foldBack2 (folder: 'T -> 'U -> 'State -> 'State) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND) (state: 'State): 'State =
        let shape = broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape
        let index = initBeforeDecrement shape
        let mutable state = state
        while decrement index shape do
            state <- folder (getBroadcastedItem index arraynd1.Tensor) (getBroadcastedItem index arraynd2.Tensor) state
        state
    let init (shape: TensorShape) (initializer: int[] -> 'T): 'T ArrayND = { Tensor = Tensor.CreateTensor(shape, initializer) }
    let isEmpty (arraynd: 'T ArrayND): bool = arraynd.Tensor.Volume = 0
    let iter (action: 'T -> unit) (arraynd: 'T ArrayND): unit = for _, x in arraynd.Tensor.Iterate() do action x
    let iter2 (action: 'T -> 'U -> unit) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): unit =
        let shape = broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape
        let index = initBeforeIncrement shape
        while increment index shape do action (getBroadcastedItem index arraynd1.Tensor) (getBroadcastedItem index arraynd2.Tensor)
    let iteri (action: int[] -> 'T -> unit) (arraynd: 'T ArrayND): unit = for i, x in arraynd.Tensor.Iterate() do action i x
    let iteri2 (action: int[] -> 'T -> 'U -> unit) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): unit =
        let shape = broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape
        let index = initBeforeIncrement shape
        while increment index shape do action index (getBroadcastedItem index arraynd1.Tensor) (getBroadcastedItem index arraynd2.Tensor)
    let map (mapping: 'T -> 'U) (arraynd: 'T ArrayND): 'U ArrayND =
        { Tensor = Tensor.CreateTensor(arraynd.Tensor.Shape, fun i -> arraynd.Tensor.GetValueNoCheck i |> mapping) }
    let map2 (mapping: 'T -> 'U -> 'V) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): 'V ArrayND =
        { Tensor = Tensor.CreateTensor(broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape, fun i ->
            mapping (getBroadcastedItem i arraynd1.Tensor) (getBroadcastedItem i arraynd2.Tensor)) }
    let map3 (mapping: 'T -> 'U -> 'V -> 'W) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND) (arraynd3: 'V ArrayND): 'W ArrayND =
        { Tensor = Tensor.CreateTensor(broadcast (broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape) arraynd3.Tensor.Shape, fun i ->
            mapping (getBroadcastedItem i arraynd1.Tensor) (getBroadcastedItem i arraynd2.Tensor) (getBroadcastedItem i arraynd3.Tensor)) }
    let mapi (mapping: int[] -> 'T -> 'U) (arraynd: 'T ArrayND): 'U ArrayND =
        { Tensor = Tensor.CreateTensor(arraynd.Tensor.Shape, fun i -> arraynd.Tensor.GetValueNoCheck i |> mapping i) }
    let mapi2 (mapping: int[] -> 'T -> 'U -> 'V) (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): 'V ArrayND =
        { Tensor = Tensor.CreateTensor(broadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape, fun i ->
            mapping i (getBroadcastedItem i arraynd1.Tensor) (getBroadcastedItem i arraynd2.Tensor)) }
    let mapFold (mapping: 'State -> 'T -> struct('Result * 'State)) (state: 'State) (arraynd: 'T ArrayND): struct('Result ArrayND * 'State) =
        let shape = arraynd.Tensor.Shape
        let mutable state = state
        let index = initBeforeIncrement shape
        let ret = Tensor shape
        while increment index shape do
            let struct(r, s) = mapping state (arraynd.Tensor.GetValueNoCheck index)
            state <- s
            ret.SetValueNoCheck(r, index)
        { Tensor = ret }, state
    let mapFoldBack (mapping: 'T -> 'State -> struct('Result * 'State)) (arraynd: 'T ArrayND) (state: 'State): struct('Result ArrayND * 'State) =
        let shape = arraynd.Tensor.Shape
        let mutable state = state
        let index = initBeforeDecrement shape
        let ret = Tensor shape
        while decrement index shape do
            let struct(r, s) = mapping (arraynd.Tensor.GetValueNoCheck index) state
            state <- s
            ret.SetValueNoCheck(r, index)
        { Tensor = ret }, state
    let max (arraynd: 'T ArrayND): 'T = arraynd |> Seq.max
    let maxBy (projection: 'T -> 'U) (arraynd: 'T ArrayND): 'T = arraynd |> Seq.maxBy projection
    let maxd (dimension: int) (arraynd: 'T ArrayND): 'T ArrayND =
        for i in dimension .. -1 .. 1 do arraynd.Tensor.Transpose(i, i-1)
        let buffer = Array.zeroCreate arraynd.Tensor.Shape.Length
        let ret = { Tensor =
            Tensor.CreateTensor(Array.tail arraynd.Tensor.Shape.shape |> TensorShape, fun j ->
                Seq.max (seq {
                    j.CopyTo(buffer, 1)
                    for i in 0 .. arraynd.Tensor.Shape[0] - 1 do
                        buffer[0] <- i
                        arraynd.Tensor.GetValueNoCheck buffer
                }))
        }
        for i in 1 .. dimension do arraynd.Tensor.Transpose(i, i-1)
        ret
    let min (arraynd: 'T ArrayND): 'T = arraynd |> Seq.min
    let minBy (projection: 'T -> 'U) (arraynd: 'T ArrayND): 'T = arraynd |> Seq.minBy projection
    let ofArray (array: 'T array): 'T ArrayND = { Tensor = Tensor.CreateVector array }
    let ofArray2D (array: 'T[,]): 'T ArrayND = { Tensor = Tensor.CreateMatrix array }
    let ofArray3D (array: 'T[,,]): 'T ArrayND = { Tensor = Tensor.CreateTensor array }
    let ofArray4D (array: 'T[,,,]): 'T ArrayND = { Tensor = Tensor.CreateTensor array }
    let ofArrayNonGeneric<'T> (array: System.Array): 'T ArrayND = { Tensor = Tensor.CreateTensor array }
    let ofList (list: 'T list): 'T ArrayND = { Tensor = Tensor.CreateTensor (list |> Array.ofList) }
    let ofSeq (source: 'T seq): 'T ArrayND = { Tensor = Tensor.CreateTensor (source |> Array.ofSeq) }
    let reduce (reduction: 'T -> 'T -> 'T) (arraynd: 'T ArrayND): 'T = Seq.reduce reduction arraynd
    let removeAt (index: int[]) (arraynd: 'T ArrayND): 'T MaskedArrayND =
        let ret = map ValueSome arraynd
        ret.Tensor[index] <- ValueNone
        ret
    let replicate (shape: TensorShape) (initial: 'T): 'T ArrayND = { Tensor = Tensor.CreateTensor(shape, fun _ -> initial) }
    let scan (scanner: 'State -> 'T -> 'State) (state: 'State) (arraynd: 'T ArrayND): 'State ArrayND =
        let mutable state = state
        let shape = arraynd.Tensor.Shape
        let ret = Tensor shape
        let index = initBeforeIncrement shape
        while increment index shape do
            state <- scanner state (arraynd.Tensor.GetValueNoCheck index)
            ret.SetValueNoCheck(state, index)
        { Tensor = ret }
    let scanBack (scanner: 'T -> 'State -> 'State) (arraynd: 'T ArrayND) (state: 'State): 'State ArrayND =
        let mutable state = state
        let shape = arraynd.Tensor.Shape
        let ret = Tensor shape
        let index = initBeforeDecrement shape
        while decrement index shape do
            state <- scanner (arraynd.Tensor.GetValueNoCheck index) state
            ret.SetValueNoCheck(state, index)
        { Tensor = ret }
    let sort (arraynd: 'T ArrayND): 'T seq = Seq.sort arraynd
    let sortBy (projection: 'T -> 'Key) (arraynd: 'T ArrayND): 'T seq = Seq.sortBy projection arraynd
    let sortDescending (arraynd: 'T ArrayND): 'T seq = Seq.sortDescending arraynd
    let sortByDescending (projection: 'T -> 'Key) (arraynd: 'T ArrayND): 'T seq = Seq.sortByDescending projection arraynd
    let sortWith (comparer: 'T -> 'T -> int) (arraynd: 'T ArrayND): 'T seq = Seq.sortWith comparer arraynd
    let inline sum (arraynd: 'T ArrayND): 'T = arraynd |> Seq.sum
    let inline sumBy (projection: 'T -> 'U) (arraynd: 'T ArrayND): 'U = arraynd |> Seq.sumBy projection
    let toArray (arraynd: 'T ArrayND): 'T array = Seq.toArray arraynd
    let toList (arraynd: 'T ArrayND): 'T list = Seq.toList arraynd
    let toSeq (arraynd: 'T ArrayND): 'T seq = arraynd
    let tryExactlyOne (arraynd: 'T ArrayND): 'T voption = if arraynd.Tensor.Volume = 1 then ValueSome (Seq.head arraynd) else ValueNone
    let exactlyOned (dimension: int) (arraynd: 'T ArrayND): 'T ArrayND voption =
        if 0 <= dimension && dimension < arraynd.Tensor.Shape.Length && arraynd.Tensor.Shape[dimension] = 1 then
            operateAlongDimension dimension arraynd.Tensor (fun tensor -> ValueSome { Tensor = tensor.GetSubtensor 0 })
        else ValueNone
    let tryFind (predicate: 'T -> bool) (arraynd: 'T ArrayND): 'T voption =
        let shape = arraynd.Tensor.Shape
        let rec loop index =
            if increment index shape then
                let x = arraynd.Tensor.GetValueNoCheck index
                if predicate x then
                    ValueSome x
                else loop index
            else ValueNone
        loop (initBeforeIncrement shape)
    let tryFindIndex (predicate: 'T -> bool) (arraynd: 'T ArrayND): int array voption =
        let shape = arraynd.Tensor.Shape
        let rec loop index =
            if increment index shape then
                if predicate (arraynd.Tensor.GetValueNoCheck index) then
                    ValueSome index
                else loop index
            else ValueNone
        loop (initBeforeIncrement shape)
    let tryFindIndexBack (predicate: 'T -> bool) (arraynd: 'T ArrayND): int array voption =
        let shape = arraynd.Tensor.Shape
        let rec loop index =
            if decrement index shape then
                if predicate (arraynd.Tensor.GetValueNoCheck index) then
                    ValueSome index
                else loop index
            else ValueNone
        loop (initBeforeDecrement shape)
    let tryItem (index: int[]) (arraynd: 'T ArrayND): 'T voption =
        if index.Length = arraynd.Tensor.Shape.Length
           && Array.forall2 (fun i s -> 0 <= i && i < s) index arraynd.Tensor.Shape.shape then
           arraynd.Tensor.GetValueNoCheck index |> ValueSome
        else ValueNone
    let tryPick (chooser: 'T -> 'U voption) (arraynd: 'T ArrayND): 'U voption =
        let mutable ret = ValueNone
        let index = initBeforeIncrement arraynd.Tensor.Shape
        while ret.IsNone && increment index arraynd.Tensor.Shape do
            ret <- chooser (arraynd.Tensor.GetValueNoCheck index)
        ret
    let updateAt (index: int[]) (value: 'T) (arraynd: 'T ArrayND): 'T ArrayND =
        let ret = arraynd.Tensor.Forward()
        ret[index] <- value
        { Tensor = ret }
    let zeroCreate (shape: TensorShape) = Tensor shape

    // Functions that depend on others
    let allPairs (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): struct('T * 'U) ArrayND = map2 (fun a b -> a, b) arraynd1 arraynd2
    let compareWith (comparer: 'T -> 'T -> int) (arraynd1: 'T ArrayND) (arraynd2: 'T ArrayND): int =
        if canBroadcast arraynd1.Tensor.Shape arraynd2.Tensor.Shape then
            map2 comparer arraynd1 arraynd2
            |> tryFind ((<>) 0)
            |> function
            | ValueSome c -> c
            | ValueNone -> compare arraynd1.Tensor.Shape.shape arraynd2.Tensor.Shape.shape
        else compare arraynd1.Tensor.Shape.shape arraynd2.Tensor.Shape.shape
    let choose (chooser: 'T -> 'U voption) (arraynd: 'T ArrayND): 'U MaskedArrayND = map chooser arraynd
    let distinct (arraynd: 'T ArrayND): 'T MaskedArrayND =
        let hashSet = System.Collections.Generic.HashSet<'T> HashIdentity.Structural<'T>
        map (fun v -> if hashSet.Add v then ValueSome v else ValueNone) arraynd
    let distinctBy (projection: 'T -> 'U) (arraynd: 'T ArrayND): 'T MaskedArrayND =
        let hashSet = System.Collections.Generic.HashSet<'U> HashIdentity.Structural<'U>
        map (fun v -> if hashSet.Add (projection v) then ValueSome v else ValueNone) arraynd
    let exactlyOne (arraynd: 'T ArrayND): 'T = Seq.exactlyOne arraynd
    let except (itemsToExclude: 'T seq) (arraynd: 'T ArrayND): 'T MaskedArrayND =
        let hashSet = System.Collections.Generic.HashSet<'T>(itemsToExclude, HashIdentity.Structural<'T>)
        map (fun v -> if hashSet.Contains v then ValueNone else ValueSome v) arraynd
    let filter (predicate: 'T -> bool) (arraynd: 'T ArrayND): 'T MaskedArrayND = map (fun x -> if predicate x then ValueSome x else ValueNone) arraynd
    let findIndex (predicate: 'T -> bool) (arraynd: 'T ArrayND): int array = tryFindIndex predicate arraynd |> ValueOption.get
    let indexed (arraynd: 'T ArrayND): struct(int[] * 'T) ArrayND = mapi (fun i x -> i.Clone() :?> int array, x) arraynd
    let permute (indexMap: int[] -> int[]) (arraynd: 'T ArrayND): 'T ArrayND =
        let res = Tensor arraynd.Tensor.Shape
        let inv = Tensor arraynd.Tensor.Shape
        arraynd |> iteri (fun i x ->
            let o = indexMap i
            inv[o] <- true
            res.SetValueNoCheck(x, o)
        )
        if Array.contains false inv.data then
            Array.permute (fun _ -> 1) [|1; 1|] |> ignore // Throw exception
        { Tensor = res }
    let pick (chooser: 'T -> 'U voption) (arraynd: 'T ArrayND): 'U = tryPick chooser arraynd |> ValueOption.get
    let rev (arraynd: 'T ArrayND): 'T ArrayND =
        let shape = arraynd.Tensor.Shape
        let res = Tensor shape
        arraynd |> iteri (fun i x ->
            res.SetValueNoCheck(x, Array.map2 (fun i s -> s - 1 - i) i shape.shape)
        )
        { Tensor = res }
    let where (predicate: 'T -> bool) (arraynd: 'T ArrayND): 'T MaskedArrayND = filter predicate arraynd
    let zip (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND): struct('T * 'U) ArrayND = map2 (fun a b -> a, b) arraynd1 arraynd2
    let zip3 (arraynd1: 'T ArrayND) (arraynd2: 'U ArrayND) (arraynd3: 'V ArrayND): struct('T * 'U * 'V) ArrayND = map3 (fun a b c -> a, b, c) arraynd1 arraynd2 arraynd3

