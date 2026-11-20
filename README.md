# FSharp.Collections.ArrayND

An F# extension over [GenericTensor](https://github.com/ASC-Community/GenericTensor) 1.0.4. It provides multidimensional arrays, elementwise arithmetic with broadcasting, maps and folds, and operations on arrays whose elements can be missing. The library targets .NET Standard 2.0. GenericTensor is restored as a NuGet dependency.

Build the library from the repository root:

```sh
dotnet build FSharp.Collections.ArrayND/FSharp.Collections.ArrayND.fsproj -c Release
```

The historical xUnit tests target .NET 6:

```sh
dotnet test FSharp.Collections.ArrayND.sln -c Release
```

The library builds with .NET SDK 10.0.403. The saved tests currently fail to compile: some assertions expect ordinary tuples or options where the library returns struct tuples or value options. The tests are included for reference.
