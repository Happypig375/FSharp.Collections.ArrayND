# FSharp.Collections.ArrayND

An F# extension over [GenericTensor](https://github.com/ASC-Community/GenericTensor) 1.0.4. It provides multidimensional arrays, broadcasting, maps and folds, and operations on arrays whose elements can be missing. The library targets .NET Standard 2.0. GenericTensor is restored as a NuGet dependency.

From the repository root, restore and build the library:

```sh
dotnet restore FSharp.Collections.ArrayND.sln
dotnet build FSharp.Collections.ArrayND/FSharp.Collections.ArrayND.fsproj -c Release --no-restore
```

The xUnit tests target .NET 6:

```sh
dotnet test FSharp.Collections.ArrayND.sln -c Release --no-restore
```

For verification on a machine with only a newer .NET runtime, the test host can use that runtime for one invocation:

```sh
DOTNET_ROLL_FORWARD=Major dotnet test FSharp.Collections.ArrayND.sln -c Release --no-restore
```

The Release library build passed with .NET SDK 10.0.403. All 166 tests passed with the process-local roll-forward to .NET 10.0.14.

An array with an empty axis keeps its declared dimensions and enumerates no elements. `max` and `maxd` ignore floating-point NaN when another value exists. An empty or all-NaN floating-point reduction returns NaN. Other empty reductions return the element type's default value. `maxd` removes the selected axis. When reducing a one-dimensional array, it returns a one-element vector because GenericTensor cannot represent a rank-zero tensor.
