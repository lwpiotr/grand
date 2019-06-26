#! /bin/bash

set -ex


test_package() {
    MYPYPATH=$PWD/user/grand/stubs mypy -m grand
    coverage run tests --unit
    python -m tests --doc
}


build_docs() {
    mkdir -p "docs/build"
    pushd "docs/build"
    git config --global user.email "travis@travis-ci.org"
    git config --global user.name "Travis CI"
    git clone "https://github.com/grand-mother/grand-docs.git" "html"
    popd

    pushd "docs"
    make html

    if [[ ! -f "build/html/.nojekyll" ]]; then
        touch "build/html/.nojekyll"
    fi
    popd
}


test_package
build_docs