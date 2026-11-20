namespace FSharp.Collections

open System
open GenericTensor.Core
open Xunit

module _Helpers =
    let assertArray (dims: TensorShape) (first: 'T) (arraynd: 'T ArrayND) =
        Assert.Equal(dims, arraynd.Tensor.Shape)
        Assert.Equal<'T>(first, arraynd.Tensor.data[0])
    let comparer = {
        new Collections.Generic.IEqualityComparer<'T> with
            override _.Equals(x, y) = x = y
            override _.GetHashCode x = hash x
    }
    let assertArrayEq (expected: 'T ArrayND) (actual: 'T ArrayND) =
        Assert.Equal(expected.Tensor.Shape, actual.Tensor.Shape)
        Assert.Equal<'T>(expected, actual, comparer)
open _Helpers
module exactlyOne =
    let check (list: 'T list) = Assert.Equal<'T>(List.exactlyOne list, list |> ArrayND.ofList |> ArrayND.exactlyOne)
    let [<Fact>] ``Built-in: Int32``() = [1] |> check
    let [<Fact>] ``Built-in: String``() = [2] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [3] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueNone] |> check
    let [<Fact>] ``Custom class: Option``() = [Some 1] |> check
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2)] |> check
    let [<Fact>] ``Throw for empty``() = Assert.Throws<ArgumentException>(fun () -> [] |> ArrayND.ofList |> ArrayND.exactlyOne |> ignore)
    let [<Fact>] ``Throw for two``() = Assert.Throws<ArgumentException>(fun () -> [1; 2] |> ArrayND.ofList |> ArrayND.exactlyOne |> ignore)
module fold =
    let check (folder: 'U -> 'T -> 'U) (state: 'U) (list: 'T list) =
        Assert.Equal<'U>(List.fold folder state list, ArrayND.fold folder state (ArrayND.ofList list))
        Assert.Equal<'U>(List.fold folder state list, ArrayND.fold folder state (ArrayND.ofArray (Array.ofList list)))
        let twoD = array2D [list; List.rev list]
        Assert.Equal<'U>(Seq.fold folder state (twoD |> Seq.cast<_>), ArrayND.fold folder state (ArrayND.ofArray2D twoD))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        Assert.Equal<'U>(Seq.fold folder state (threeD |> Seq.cast<_>), ArrayND.fold folder state (ArrayND.ofArray3D threeD))
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check (+) 3
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check (sprintf "%s%s") "3"
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check (fun a c -> a + int c) 2
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check (ValueOption.map2 (-)) (ValueSome 3)
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check (fun _ _ -> 0) 2
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check (fun a (x, _) -> a + x) 2
module indexed =
    let check (list: 'T list) =
        assertArrayEq (List.indexed list |> List.map (fun (i, x) -> struct ([|i|], x)) |> ArrayND.ofList) (ArrayND.indexed (ArrayND.ofList list))
        assertArrayEq (List.indexed list |> List.map (fun (i, x) -> struct ([|i|], x)) |> ArrayND.ofList) (ArrayND.indexed (ArrayND.ofArray (Array.ofList list)))
        let twoD = array2D [list; List.rev list]
        assertArrayEq (Array2D.mapi (fun i j x -> struct ([|i; j|], x)) twoD |> ArrayND.ofArray2D) (ArrayND.indexed (ArrayND.ofArray2D twoD))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        assertArrayEq (Array3D.mapi (fun i j k x -> struct ([|i; j; k|], x)) threeD |> ArrayND.ofArray3D) (ArrayND.indexed (ArrayND.ofArray3D threeD))
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check
module map =
    let check (map: 'T -> 'U) (list: 'T list) =
        assertArrayEq (List.map map list |> ArrayND.ofList) (ArrayND.map map (ArrayND.ofList list))
        assertArrayEq (List.map map list |> ArrayND.ofList) (ArrayND.map map (ArrayND.ofArray (Array.ofList list)))
        let twoD = array2D [list; List.rev list]
        assertArrayEq (Array2D.map map twoD |> ArrayND.ofArray2D) (ArrayND.map map (ArrayND.ofArray2D twoD))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        assertArrayEq (Array3D.map map threeD |> ArrayND.ofArray3D) (ArrayND.map map (ArrayND.ofArray3D threeD))
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check ((+) 1)
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check ((+) "prefix")
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check ((+) '\002')
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check (ValueOption.map ((+) 1))
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check (Option.map ((+) 1))
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check (fun _ -> 3)
module map2 =
    let other = [1 .. 10]
    let check (map: 'T -> int -> 'U) (list: 'T list) =
        assertArrayEq (List.map2 map list other |> ArrayND.ofList) (ArrayND.map2 map (ArrayND.ofList list) (ArrayND.ofList other))
        assertArrayEq (List.map2 map list other |> ArrayND.ofList) (ArrayND.map2 map (ArrayND.ofArray (Array.ofList list)) (ArrayND.ofList other))
        // test broadcasting
        let twoD = array2D [list; List.rev list]
        let resultTwoD = Array2D.init 2 list.Length (fun i j -> map twoD[i, j] other[j])
        assertArrayEq (resultTwoD |> ArrayND.ofArray2D) (ArrayND.map2 map (ArrayND.ofArray2D twoD) (ArrayND.ofList other))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        let resultThreeD = Array3D.init list.Length list.Length list.Length (fun _ j k -> map list[j] other[k])
        assertArrayEq (resultThreeD |> ArrayND.ofArray3D) (ArrayND.map2 map (ArrayND.ofArray3D threeD) (ArrayND.ofList other))
    let [<Fact>] ``Built-in: Int32``() = [31 .. 40] |> check (+)
    let [<Fact>] ``Built-in: String``() = [31 .. 40] |> List.map string |> check (sprintf "%s%dsuffix")
    let [<Fact>] ``Custom struct: Char``() = [31 .. 40] |> List.map char |> check (fun c i -> c + char i |> int64)
    let [<Fact>] ``Custom struct: ValueOption``() = [31 .. 40] |> List.map ValueSome |> check (fun v i -> ValueOption.map ((+) i) v)
    let [<Fact>] ``Custom class: Option``() = [31 .. 40] |> List.map Some |> check (fun o i -> Option.map ((-) i) o)
    let [<Fact>] ``Custom class: Tuple``() = [31 .. 40] |> List.mapi (fun i x -> x, i) |> check (fun (x, y) i -> struct(x-i, y-i))
module map3 =
    let other = [1 .. 10]
    let other21D = [for i in 10 .. -1 .. 1 do char i]
    let other2 = array2D [for c in other21D do [c]]
    let check (map: 'T -> int -> char -> 'U) (list: 'T list) =
        assertArrayEq (List.map3 map list other other21D |> ArrayND.ofList) (ArrayND.map3 map (ArrayND.ofList list) (ArrayND.ofList other) (ArrayND.ofSeq other21D))
        assertArrayEq (List.map3 map list other other21D |> ArrayND.ofList) (ArrayND.map3 map (ArrayND.ofArray (Array.ofList list)) (ArrayND.ofList other) (ArrayND.ofList other21D))
        // test broadcasting
        let twoD = array2D [for _ in 1 .. 2 .. 10 do list; List.rev list]
        let resultTwoD = Array2D.init 10 list.Length (fun i j -> map twoD[i, j] other[j] other2[i, 0])
        assertArrayEq (resultTwoD |> ArrayND.ofArray2D) (ArrayND.map3 map (ArrayND.ofArray2D twoD) (ArrayND.ofList other) (ArrayND.ofArray2D other2))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        let resultThreeD = Array3D.init list.Length list.Length list.Length (fun _ j k -> map list[j] other[k] other2[j, 0])
        assertArrayEq (resultThreeD |> ArrayND.ofArray3D) (ArrayND.map3 map (ArrayND.ofArray3D threeD) (ArrayND.ofList other) (ArrayND.ofArray2D other2))
    let [<Fact>] ``Built-in: Int32``() = [31 .. 40] |> check (fun a b c -> a + b + int c)
    let [<Fact>] ``Built-in: String``() = [31 .. 40] |> List.map string |> check (sprintf "%s%dsuffix%c")
    let [<Fact>] ``Custom struct: Char``() = [31 .. 40] |> List.map char |> check (fun c i j -> c + char i + j |> int64)
    let [<Fact>] ``Custom struct: ValueOption``() = [31 .. 40] |> List.map ValueSome |> check (fun v i c -> ValueOption.map ((+) (i + int c)) v)
    let [<Fact>] ``Custom class: Option``() = [31 .. 40] |> List.map Some |> check (fun o i c -> $"{Option.map ((-) i) o}{c}")
    let [<Fact>] ``Custom class: Tuple``() = [31 .. 40] |> List.mapi (fun i x -> x, i) |> check (fun (x, y) i c -> struct(c, x-i, y-i))
module mapi =
    let check (map: int[] -> 'T -> 'U) (list: 'T list) =
        assertArrayEq (List.mapi (fun i -> map [|i|]) list |> ArrayND.ofList) (ArrayND.mapi map (ArrayND.ofList list))
        assertArrayEq (List.mapi (fun i -> map [|i|]) list |> ArrayND.ofList) (ArrayND.mapi map (ArrayND.ofArray (Array.ofList list)))
        let twoD = array2D [list; List.rev list]
        assertArrayEq (Array2D.mapi (fun i j -> map [|i; j|]) twoD |> ArrayND.ofArray2D) (ArrayND.mapi map (ArrayND.ofArray2D twoD))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        assertArrayEq (Array3D.mapi (fun i j k -> map [|i; j; k|]) threeD |> ArrayND.ofArray3D) (ArrayND.mapi map (ArrayND.ofArray3D threeD))
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check (sprintf "%A%i")
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check (sprintf "%A%s")
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check (sprintf "%A%c")
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check (sprintf "%A%A")
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check (sprintf "%A%A")
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check (sprintf "%A%A")
module mapi2 =
    let other = [1 .. 10]
    let check (map: int[] -> 'T -> int -> 'U) (list: 'T list) =
        assertArrayEq (List.mapi2 (fun i -> map [|i|]) list other |> ArrayND.ofList) (ArrayND.mapi2 map (ArrayND.ofList list) (ArrayND.ofList other))
        assertArrayEq (List.mapi2 (fun i -> map [|i|]) list other |> ArrayND.ofList) (ArrayND.mapi2 map (ArrayND.ofArray (Array.ofList list)) (ArrayND.ofList other))
        // test broadcasting
        let twoD = array2D [list; List.rev list]
        let resultTwoD = Array2D.init 2 list.Length (fun i j -> map [|i; j|] twoD[i, j] other[j])
        assertArrayEq (resultTwoD |> ArrayND.ofArray2D) (ArrayND.mapi2 map (ArrayND.ofArray2D twoD) (ArrayND.ofList other))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        let resultThreeD = Array3D.init list.Length list.Length list.Length (fun i j k -> map [|i; j; k|] list[j] other[k])
        assertArrayEq (resultThreeD |> ArrayND.ofArray3D) (ArrayND.mapi2 map (ArrayND.ofArray3D threeD) (ArrayND.ofList other))
    let [<Fact>] ``Built-in: Int32``() = [31 .. 40] |> check (sprintf "%A%d%d")
    let [<Fact>] ``Built-in: String``() = [31 .. 40] |> List.map string |> check (sprintf "%A%s%d")
    let [<Fact>] ``Custom struct: Char``() = [31 .. 40] |> List.map char |> check (sprintf "%A%c%d")
    let [<Fact>] ``Custom struct: ValueOption``() = [31 .. 40] |> List.map ValueSome |> check (sprintf "%A%A%d")
    let [<Fact>] ``Custom class: Option``() = [31 .. 40] |> List.map Some |> check (sprintf "%A%A%d")
    let [<Fact>] ``Custom class: Tuple``() = [31 .. 40] |> List.mapi (fun i x -> x, i) |> check (sprintf "%A%A%d")
module max =
    let check (list: 'T list) =
        Assert.Equal<'T>(List.max list, ArrayND.max(ArrayND.ofList list))
        Assert.Equal<'T>(List.max list, ArrayND.max(ArrayND.ofArray (Array.ofList list)))
        Assert.Equal<'T>(List.max list, ArrayND.max(ArrayND.ofArray2D (array2D [list; List.rev list])))
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check
    let [<Fact>] ``NaN``() = [1.; 3.; nan; 2.] |> check
    let [<Fact>] ``Single value``() = Assert.Equal(1., ArrayND.ofSeq [1.] |> ArrayND.max)
    let [<Fact>] ``No value``() =
        Assert.Equal(0, ArrayND.ofSeq [] |> ArrayND.max)
        Assert.Equal(nan, ArrayND.ofSeq [] |> ArrayND.max)
module maxd =
    let check (map: int -> 'T) =
        let input = Array2D.init 4 4 (fun i j -> map (i * 4 + j))
        let expectd0 = Array.init 4 (fun i -> input[*, i] |> Array.filter (nan.Equals >> not) |> Array.max)
        let expectd1 = Array.init 4 (fun i -> input[i, *] |> Array.filter (nan.Equals >> not) |> Array.max)
        input |> ArrayND.ofArray2D |> ArrayND.maxd 0 |> assertArrayEq (ArrayND.ofArray expectd0)
        input |> ArrayND.ofArray2D |> ArrayND.maxd 1 |> assertArrayEq (ArrayND.ofArray expectd1)
        let input = Array3D.init 4 4 4 (fun i j k -> map (i * 4 * 4 + j * 4 + k))
        let expectd0 = Array2D.init 4 4 (fun i j -> input[*, i, j] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        let expectd1 = Array2D.init 4 4 (fun i j -> input[i, *, j] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        let expectd2 = Array2D.init 4 4 (fun i j -> input[i, j, *] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        input |> ArrayND.ofArray3D |> ArrayND.maxd 0 |> assertArrayEq (ArrayND.ofArray2D expectd0)
        input |> ArrayND.ofArray3D |> ArrayND.maxd 1 |> assertArrayEq (ArrayND.ofArray2D expectd1)
        input |> ArrayND.ofArray3D |> ArrayND.maxd 2 |> assertArrayEq (ArrayND.ofArray2D expectd2)
        let input = Array4D.init 4 4 4 4 (fun i j k l -> map (i * 4 * 4 * 4 + j * 4 * 4 + k * 4 + l))
        let expectd0 = Array3D.init 4 4 4 (fun i j k -> input[*, i, j, k] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        let expectd1 = Array3D.init 4 4 4 (fun i j k -> input[i, *, j, k] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        let expectd2 = Array3D.init 4 4 4 (fun i j k -> input[i, j, *, k] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        let expectd3 = Array3D.init 4 4 4 (fun i j k -> input[i, j, k, *] |> Seq.cast<'T> |> Seq.filter (nan.Equals >> not) |> Seq.max)
        input |> ArrayND.ofArray4D |> ArrayND.maxd 0 |> assertArrayEq (ArrayND.ofArray3D expectd0)
        input |> ArrayND.ofArray4D |> ArrayND.maxd 1 |> assertArrayEq (ArrayND.ofArray3D expectd1)
        input |> ArrayND.ofArray4D |> ArrayND.maxd 2 |> assertArrayEq (ArrayND.ofArray3D expectd2)
        input |> ArrayND.ofArray4D |> ArrayND.maxd 3 |> assertArrayEq (ArrayND.ofArray3D expectd3)
    let [<Fact>] ``Built-in: Int32``() = check id
    let [<Fact>] ``Built-in: String``() = check string
    let [<Fact>] ``Custom struct: Char``() = check char
    let [<Fact>] ``Custom struct: ValueOption``() = check (function i when i % 3 <> 1 -> ValueSome i | _ -> ValueNone)
    let [<Fact>] ``Custom class: Option``() = check (function i when i % 3 <> 1 -> Some i | _ -> None)
    let [<Fact>] ``Custom class: Tuple``() = check (fun i -> i+i, i)
    let [<Fact>] ``NaN``() = check (function i when i % 3 <> 0 -> float i | _ -> nan)
    let [<Fact>] ``Single value``() = Assert.Equal(1., ArrayND.ofSeq [1.] |> ArrayND.maxd 0 |> ArrayND.exactlyOne)
    let [<Fact>] ``No value``() =
        Assert.Equal(0, ArrayND.ofSeq [] |> ArrayND.maxd 0 |> ArrayND.exactlyOne)
        Assert.Equal(nan, ArrayND.ofSeq [] |> ArrayND.maxd 0 |> ArrayND.exactlyOne)
module maxBy =
    let order = [|5; 2; 9; 3; 1; 0; 0; 4; 8; 6; 7|]
    let order2 = [|Some 5; Some 2; None; None; None; Some 0; Some 0; None; Some 8; Some 6; Some 7|]
    let check' projection (list: 'T list) =
        Assert.Equal<'T>(List.maxBy projection list, ArrayND.maxBy projection (ArrayND.ofList list))
        Assert.Equal<'T>(List.maxBy projection list, ArrayND.maxBy projection (ArrayND.ofArray (Array.ofList list)))
        Assert.Equal<'T>(List.maxBy projection list, ArrayND.maxBy projection (ArrayND.ofArray2D (array2D [list; List.rev list])))
    let check projection (list: 'T list) =
        check' (fun n -> order[projection n]) list // Built-in
        check' (fun n -> order2[projection n]) list // Not built-in
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check id
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check int
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check int
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 2] |> check (ValueOption.defaultValue 0)
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 2] |> check (Option.defaultValue 0)
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check (fun (x, y) -> x + y)
    let [<Fact>] ``NaN``() = [1.; 3.; nan; 2.] |> check' id
    let [<Fact>] ``Single value``() = Assert.Equal(1., ArrayND.ofSeq [1] |> ArrayND.maxBy (Random().Next))
    let [<Fact>] ``No value``() =
        Assert.ThrowsAny<exn>(fun () -> ArrayND.ofSeq [] |> ArrayND.maxBy (fun _ -> 2) |> box<int>) |> ignore
        Assert.ThrowsAny<exn>(fun () -> ArrayND.ofSeq [] |> ArrayND.maxBy (fun _ -> 2) |> box<int list>)
module min =
    let check (list: 'T list) =
        Assert.Equal<'T>(List.min list, ArrayND.min(ArrayND.ofList list))
        Assert.Equal<'T>(List.min list, ArrayND.min(ArrayND.ofArray (Array.ofList list)))
        Assert.Equal<'T>(List.min list, ArrayND.min(ArrayND.ofArray2D (array2D [list; List.rev list])))
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check
    let [<Fact>] ``NaN``() = [1.; 3.; nan; 2.] |> check
    let [<Fact>] ``Single value``() = Assert.Equal(1., ArrayND.ofSeq [1.] |> ArrayND.min)
    let [<Fact>] ``No value``() = Assert.Throws<ArgumentException>(fun () -> ArrayND.ofSeq [] |> ArrayND.min)
module minBy =
    let order = [|5; 2; 9; 3; 1; 0; 0; 4; 8; 6; 7|]
    let order2 = [|Some 5; Some 2; None; None; None; Some 0; Some 0; None; Some 8; Some 6; Some 7|]
    let check' projection (list: 'T list) =
        Assert.Equal<'T>(List.minBy projection list, ArrayND.minBy projection (ArrayND.ofList list))
        Assert.Equal<'T>(List.minBy projection list, ArrayND.minBy projection (ArrayND.ofArray (Array.ofList list)))
        Assert.Equal<'T>(List.minBy projection list, ArrayND.minBy projection (ArrayND.ofArray2D (array2D [list; List.rev list])))
    let check projection (list: 'T list) =
        check' (fun n -> order[projection n]) list // Built-in
        check' (fun n -> order2[projection n]) list // Not built-in
    let [<Fact>] ``Built-in: Int32``() = [1 .. 10] |> check id
    let [<Fact>] ``Built-in: String``() = [1 .. 10] |> List.map string |> check int
    let [<Fact>] ``Custom struct: Char``() = [1 .. 10] |> List.map char |> check int
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueSome 1; ValueNone; ValueSome 0] |> check (ValueOption.defaultValue 0)
    let [<Fact>] ``Custom class: Option``() = [Some 1; None; Some 0] |> check (Option.defaultValue 0)
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2); (2, 3); (3, 4)] |> check (fun (x, y) -> x + y)
    let [<Fact>] ``NaN``() = [1.; 3.; nan; 2.] |> check' id
    let [<Fact>] ``Single value``() = Assert.Equal(1., ArrayND.ofSeq [1] |> ArrayND.minBy (Random().Next))
    let [<Fact>] ``No value``() = Assert.Throws<ArgumentException>(fun () -> ArrayND.ofSeq [] |> ArrayND.minBy (fun () -> 2))
module ofArray =
    let check (source: 'T array) assign =
        let a = ArrayND.ofArray source
        let head = Array.head source
        assertArray (TensorShape [|3|]) head a
        source[0] <- assign
        assertArray (TensorShape [|3|]) head a
    let [<Fact>] ``Built-in: Int32`` () = check [|1; 2; 3|] 9
    let [<Fact>] ``Built-in: String`` () = check [|"1"; "2"; "3"|] "9"
    let [<Fact>] ``Custom struct: Char`` () = check [|'1'; '2'; '3'|] '9'
    let [<Fact>] ``Custom struct: ValueOption`` () = check [|ValueSome 1; ValueSome 2; ValueNone|] (ValueSome 3)
    let [<Fact>] ``Custom class: Option`` () = check [|Some 1; Some 2; None|] None
    let [<Fact>] ``Custom class: Tuple`` () = check [|(1, 2); (3, 4); (5, 6)|] (8, 9)
    let [<Fact>] ``Empty struct`` () = Assert.Equal(TensorShape[|0|], (ArrayND.ofArray Array.empty<char>).Tensor.Shape)
    let [<Fact>] ``Empty class`` () = Assert.Equal(TensorShape[|0|], (ArrayND.ofArray Array.empty<int list>).Tensor.Shape)
module ofArray2D =
    let check (line1: 'T list) (line2: 'T list) dtype assign =
        let source = array2D [line1; line2]
        let a = ArrayND.ofArray2D source
        assertArray (TensorShape[|2; 2|]) source[0,0] a
        source[0,0] <- assign
        assertArray (TensorShape[|2; 2|]) source[0,0] a
    let [<Fact>] ``Built-in: Int32`` () = check [1; 2] [3; 3] 9
    let [<Fact>] ``Built-in: String`` () = check ["1"; "2"] ["3"; "3"] "9"
    let [<Fact>] ``Custom struct: Char`` () = check ['1'; '2'] ['3'; '3'] '9'
    let [<Fact>] ``Custom struct: ValueOption`` () = check [ValueSome 1; ValueSome 2] [ValueNone; ValueNone] ValueNone
    let [<Fact>] ``Custom class: Option`` () = check [Some 1; Some 2] [None; None] None
    let [<Fact>] ``Custom class: Tuple`` () = check [(1, 2); (3, 4)] [(5, 6); (5, 6)] (8, 9)
    let [<Fact>] ``Empty struct`` () = Assert.Equal(TensorShape[|0; 0|], (ArrayND.ofArray2D (array2D [] : char[,])).Tensor.Shape)
    let [<Fact>] ``Empty class`` () = Assert.Equal(TensorShape[|0; 0|], (ArrayND.ofArray2D (array2D [] : int list[,])).Tensor.Shape)
module ofArray3D =
    let check map =
        let source = Array3D.init 4 3 2 (fun a b c -> a * 3 * 2 + b * 2 + c |> map)
        let a = ArrayND.ofArray3D source
        let check = assertArray (TensorShape [|4; 3; 2|]) source[0,0,0]
        check a
        source[0,0,0] <- map 9
        check a
    let [<Fact>] ``Built-in: Int32`` () = check id
    let [<Fact>] ``Custom struct: Char`` () = check char
    let [<Fact>] ``Custom class: Option`` () = check Some
    let [<Fact>] ``Empty struct`` () = Assert.Equal(TensorShape[|0; 2; 0|], (ArrayND.ofArray3D (Array3D.zeroCreate 0 2 0 : char[,,])).Tensor.Shape)
    let [<Fact>] ``Empty class`` () = Assert.Equal(TensorShape[|0; 2; 0|], (ArrayND.ofArray3D (Array3D.zeroCreate 0 2 0 : int list[,,])).Tensor.Shape)
module ofArray4D =
    let check map =
        let source = Array4D.init 4 3 2 1 (fun a b c _ -> a * 3 * 2 + b * 2 + c |> map)
        let a = ArrayND.ofArray4D source
        let check = assertArray (TensorShape [|4; 3; 2; 1|]) source[0,0,0,0]
        check a
        source[0,0,0,0] <- map 9
        check a
    let [<Fact>] ``Built-in: Int32`` () = check id
    let [<Fact>] ``Custom struct: Char`` () = check char
    let [<Fact>] ``Custom class: Option`` () = check Some
    let [<Fact>] ``Empty struct`` () = Assert.Equal(TensorShape[|0; 0; 0; 4|], (ArrayND.ofArray4D (Array4D.zeroCreate 0 0 0 4 : char[,,,])).Tensor.Shape)
    let [<Fact>] ``Empty class`` () = Assert.Equal(TensorShape[|0; 0; 0; 4|], (ArrayND.ofArray4D (Array4D.zeroCreate 0 0 0 4 : int list[,,,])).Tensor.Shape)
module ofArrayNonGeneric =
    let check8D (assign: 'T) =
        let source = System.Array.CreateInstance(typeof<'T>, [|9; 1; 2; 1; 1; 1; 1; 2|])
        let a = ArrayND.ofArrayNonGeneric<'T> source
        let check = assertArray (TensorShape [|9; 1; 2; 1; 1; 1; 1; 2|]) (source |> Seq.cast<'T> |> Seq.head)
        check a
        source.SetValue(assign, Array.replicate 8 0)
        check a
    let [<Fact>] ``8D Built-in: Int32`` () = check8D 9
    let [<Fact>] ``8D Custom struct: Char`` () = check8D '\009'
    let [<Fact>] ``8D Custom class: Option`` () = check8D (Some 9)
module ofList =
    let [<Fact>] ``Built-in: Int32`` () =
        ArrayND.ofList [3 .. 7] |> assertArray (TensorShape [|5|]) 3
    let [<Fact>] ``Custom struct: Char`` () =
        ArrayND.ofList ['3' .. '7'] |> assertArray (TensorShape [|5|]) '3'
    let [<Fact>] ``Custom class: Option`` () =
        ArrayND.ofList (['3' .. '7'] |> List.map Some) |> assertArray (TensorShape [|5|]) (Some '3')
module ofSeq =
    let [<Fact>] ``Built-in: Int32`` () =
        ArrayND.ofSeq (seq { 3 .. 7 }) |> assertArray (TensorShape [|5|]) 3
    let [<Fact>] ``Custom struct: Char`` () =
        ArrayND.ofSeq (seq { '3' .. '7' }) |> assertArray (TensorShape [|5|]) '3'
    let [<Fact>] ``Custom class: Option`` () =
        ArrayND.ofSeq (['3' .. '7'] |> List.map Some) |> assertArray (TensorShape [|5|]) (Some '3')
module sum =
    let inline check (map: int -> 'T) =
        let input = List.map map [1 .. 10]
        Assert.Equal<'T>(List.sum input, ArrayND.sum (ArrayND.ofList input))
        Assert.Equal<'T>(List.sum input + List.sum input, ArrayND.sum (ArrayND.ofArray2D (array2D [input; List.rev input])))
    let inline check' (map: int -> 'T) (res: 'T) =
        let input = List.map map [1 .. 10]
        Assert.Equal<'T>(res, ArrayND.sum (ArrayND.ofList input))
        Assert.Equal<'T>(res + res + res, ArrayND.sum (ArrayND.ofArray2D (array2D [input; input; input])))
    let [<Fact>] ``Built-in: Int8`` () = check int8
    let [<Fact>] ``Built-in: UInt8`` () = check uint8
    let [<Fact>] ``Built-in: Int16`` () = check int16
    let [<Fact>] ``Built-in: UInt16`` () = check uint16
    let [<Fact>] ``Built-in: Int32`` () = check id
    let [<Fact>] ``Built-in: UInt32`` () = check uint32
    let [<Fact>] ``Built-in: Int64`` () = check int64
    let [<Fact>] ``Built-in: UInt64`` () = check uint64
    let [<Fact>] ``Built-in: Float32`` () = check float32
    let [<Fact>] ``Built-in: Float`` () = check float
    let [<Fact>] ``Built-in: Decimal`` () = check decimal
    let [<Fact>] ``Built-in: BigInt`` () = check bigint
    let [<Fact>] ``Sums to int32: Char`` () = check char
module tryExactlyOne =
    let check (list: 'T list) =
        Assert.Equal<'T voption>(List.tryExactlyOne list |> ValueOption.ofOption, list |> ArrayND.ofList |> ArrayND.tryExactlyOne)
        let list = [yield! list; yield! list]
        Assert.Equal<'T voption>(List.tryExactlyOne list |> ValueOption.ofOption, list |> ArrayND.ofList |> ArrayND.tryExactlyOne)
        let list = [if false then yield! list]
        Assert.Equal<'T voption>(List.tryExactlyOne list |> ValueOption.ofOption, list |> ArrayND.ofList |> ArrayND.tryExactlyOne)
    let [<Fact>] ``Built-in: Int32``() = [1] |> check
    let [<Fact>] ``Built-in: String``() = [2] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [3] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [ValueNone] |> check
    let [<Fact>] ``Custom class: Option``() = [Some 1] |> check
    let [<Fact>] ``Custom class: Tuple``() = [(1, 2)] |> check
module zip =
    let other = [1 .. 10]
    let check (list: 'T list) =
        assertArrayEq (List.map2 (fun x y -> struct (x, y)) list other |> ArrayND.ofList) (ArrayND.zip (ArrayND.ofList list) (ArrayND.ofList other))
        assertArrayEq (List.map2 (fun x y -> struct (x, y)) list other |> ArrayND.ofList) (ArrayND.zip (ArrayND.ofArray (Array.ofList list)) (ArrayND.ofList other))
        // test broadcasting
        let twoD = array2D [list; List.rev list]
        let resultTwoD = Array2D.init 2 list.Length (fun i j -> struct (twoD[i, j], other[j]))
        assertArrayEq (resultTwoD |> ArrayND.ofArray2D) (ArrayND.zip (ArrayND.ofArray2D twoD) (ArrayND.ofList other))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        let resultThreeD = Array3D.init list.Length list.Length list.Length (fun _ j k -> struct (list[j], other[k]))
        assertArrayEq (resultThreeD |> ArrayND.ofArray3D) (ArrayND.zip (ArrayND.ofArray3D threeD) (ArrayND.ofList other))
    let [<Fact>] ``Built-in: Int32``() = [31 .. 40] |> check
    let [<Fact>] ``Built-in: String``() = [31 .. 40] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [31 .. 40] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [31 .. 40] |> List.map ValueSome |> check
    let [<Fact>] ``Custom class: Option``() = [31 .. 40] |> List.map Some |> check
    let [<Fact>] ``Custom class: Tuple``() = [31 .. 40] |> List.mapi (fun i x -> x, i) |> check
module zip3 =
    let other = [1 .. 10]
    let other21D = [for i in 10 .. -1 .. 1 do char i]
    let other2 = array2D [for c in other21D do [c]]
    let check (list: 'T list) =
        assertArrayEq (List.map3 (fun x y z -> struct (x, y, z)) list other other21D |> ArrayND.ofList) (ArrayND.zip3 (ArrayND.ofList list) (ArrayND.ofList other) (ArrayND.ofSeq other21D))
        assertArrayEq (List.map3 (fun x y z -> struct (x, y, z)) list other other21D |> ArrayND.ofList) (ArrayND.zip3 (ArrayND.ofArray (Array.ofList list)) (ArrayND.ofList other) (ArrayND.ofList other21D))
        // test broadcasting
        let twoD = array2D [for _ in 1 .. 2 .. 10 do list; List.rev list]
        let resultTwoD = Array2D.init 10 list.Length (fun i j -> struct (twoD[i, j], other[j], other2[i, 0]))
        assertArrayEq (resultTwoD |> ArrayND.ofArray2D) (ArrayND.zip3 (ArrayND.ofArray2D twoD) (ArrayND.ofList other) (ArrayND.ofArray2D other2))
        let threeD = Array3D.init list.Length list.Length 1 (fun _ j _ -> list[j])
        let resultThreeD = Array3D.init list.Length list.Length list.Length (fun _ j k -> struct (list[j], other[k], other2[j, 0]))
        assertArrayEq (resultThreeD |> ArrayND.ofArray3D) (ArrayND.zip3 (ArrayND.ofArray3D threeD) (ArrayND.ofList other) (ArrayND.ofArray2D other2))
    let [<Fact>] ``Built-in: Int32``() = [31 .. 40] |> check
    let [<Fact>] ``Built-in: String``() = [31 .. 40] |> List.map string |> check
    let [<Fact>] ``Custom struct: Char``() = [31 .. 40] |> List.map char |> check
    let [<Fact>] ``Custom struct: ValueOption``() = [31 .. 40] |> List.map ValueSome |> check
    let [<Fact>] ``Custom class: Option``() = [31 .. 40] |> List.map Some |> check
    let [<Fact>] ``Custom class: Tuple``() = [31 .. 40] |> List.mapi (fun i x -> x, i) |> check