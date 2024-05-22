{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [
      clang-tools
      cmake
      gcc13

      (pkgs.python3.withPackages (python-pkgs: [
        python-pkgs.pip
        python-pkgs.scipy
        python-pkgs.python-lsp-server
        python-pkgs.pylsp-mypy
        python-pkgs.pandas
        python-pkgs.pandas-stubs
        python-pkgs.matplotlib
        (python-pkgs.buildPythonPackage rec {
          name = "SciencePlots";
          src = fetchFromGitHub {
            owner = "garrettj403";
            repo = "SciencePlots";
            rev = "0064c84dfce1b9b420783331fb674d310e921e36";
            sha256 = "1vmpfk1n2zjj112l0xdw3k5p0h8m0ypjxclzmlwdhjp2vl4ixm3y";
          };
          propagatedBuildInputs = [ python-pkgs.matplotlib ];
        })
      ]))
    ];
}
