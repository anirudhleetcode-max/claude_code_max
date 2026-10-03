from __future__ import annotations

import textwrap

from ai_engineer.repo.symbols import Symbol, extract_imports, extract_symbols


def by_name(symbols: list[Symbol]) -> dict[str, Symbol]:
    return {s.name: s for s in symbols}


def kinds(symbols: list[Symbol]) -> set[tuple[str, str, str | None]]:
    return {(s.kind, s.name, s.parent) for s in symbols}


PY_SOURCE = textwrap.dedent(
    """\
    import os
    MAX_ITEMS = 10
    lower_case = 3

    class Engine(Base):
        LIMIT = 5

        def run(self, x: int = 1) -> int:
            def inner():
                pass
            return x

        async def stop(self):
            ...

        class Config:
            def load(self):
                pass

    def start_engine(*args, **kwargs) -> "Engine":
        return Engine()

    if True:
        def conditional():
            pass
    """
)


def test_python_symbols_via_ast() -> None:
    symbols = extract_symbols("pkg/core.py", PY_SOURCE, "python")
    assert kinds(symbols) == {
        ("const", "MAX_ITEMS", None),
        ("class", "Engine", None),
        ("method", "run", "Engine"),
        ("method", "stop", "Engine"),
        ("class", "Config", "Engine"),
        ("method", "load", "Config"),
        ("function", "start_engine", None),
        ("function", "conditional", None),
    }
    named = by_name(symbols)
    assert named["Engine"].line == 5
    assert named["Engine"].end_line == 18
    assert named["run"].signature == "def run(self, x: int=1) -> int"
    assert named["stop"].signature.startswith("async def stop")
    assert named["Engine"].signature == "class Engine(Base)"
    assert [s.line for s in symbols] == sorted(s.line for s in symbols)


def test_python_symbols_regex_fallback_on_syntax_error() -> None:
    broken = "class Foo:\n    def bar(self):\n        pass\ndef top(:\n    pass\nLIMIT = 3\n"
    symbols = extract_symbols("x.py", broken, "python")
    assert kinds(symbols) == {
        ("class", "Foo", None),
        ("method", "bar", "Foo"),
        ("function", "top", None),
        ("const", "LIMIT", None),
    }
    assert by_name(symbols)["top"].line == 4


TS_SOURCE = textwrap.dedent(
    """\
    import { thing } from './thing';
    export interface Props {
      name: string;
    }
    export type Id = string | number;
    export enum Color { Red, Green }
    export default async function main(argv: string[]): Promise<void> {
      function nested() {}
      if (argv) { run(); }
    }
    export const handler = async (req: Request) => {
      return 1;
    };
    const helper = function () { return 2; };
    export const API_URL = "https://example.invalid";
    const internal = 3;
    export class Service<T> extends Base {
      private count = 0;
      static create(): Service<number> {
        return new Service();
      }
      constructor(private readonly dep: Dep) {
        super();
      }
      async fetch(id: string): Promise<T> {
        for (const x of this.items) { process(x); }
        return null as any;
      }
      onClick = (event: Event) => {
        this.count++;
      };
    }
    describe("suite", () => {
      it("works", () => {});
      function localHelper() {}
    });
    """
)


def test_typescript_symbols() -> None:
    symbols = extract_symbols("src/service.ts", TS_SOURCE, "typescript")
    assert kinds(symbols) == {
        ("interface", "Props", None),
        ("type", "Id", None),
        ("enum", "Color", None),
        ("function", "main", None),
        ("function", "handler", None),
        ("function", "helper", None),
        ("const", "API_URL", None),
        ("class", "Service", None),
        ("method", "create", "Service"),
        ("method", "constructor", "Service"),
        ("method", "fetch", "Service"),
        ("method", "onClick", "Service"),
    }
    named = by_name(symbols)
    assert (named["main"].line, named["main"].end_line) == (7, 10)
    assert (named["handler"].line, named["handler"].end_line) == (11, 13)
    assert (named["Service"].line, named["Service"].end_line) == (17, 32)
    assert named["fetch"].end_line == 28


def test_javascript_commonjs_and_class_methods() -> None:
    source = textwrap.dedent(
        """\
        const fs = require('fs');
        function readConfig(path) {
          return fs.readFileSync(path);
        }
        module.exports.loadAll = function () {};
        exports.parse = (text) => JSON.parse(text);
        class Cache {
          get(key) { return this.map[key]; }
          set(key, value) {
            if (key) { this.map[key] = value; }
          }
        }
        """
    )
    assert kinds(extract_symbols("lib/config.js", source, "javascript")) == {
        ("function", "readConfig", None),
        ("function", "loadAll", None),
        ("function", "parse", None),
        ("class", "Cache", None),
        ("method", "get", "Cache"),
        ("method", "set", "Cache"),
    }


def test_braces_inside_strings_and_comments_do_not_confuse_scanner() -> None:
    source = textwrap.dedent(
        """\
        class A {
          a() { const s = "}"; const t = '{'; /* } */ // }
            return `${s}}`;
          }
          b() {}
        }
        function after() {}
        """
    )
    assert kinds(extract_symbols("a.js", source, "javascript")) == {
        ("class", "A", None),
        ("method", "a", "A"),
        ("method", "b", "A"),
        ("function", "after", None),
    }


GO_SOURCE = textwrap.dedent(
    """\
    package store

    import (
    \t"fmt"
    \tutil "example.com/demo/internal/util"
    )

    const Version = "1"

    type Store struct {
    \titems map[string]string
    }

    type Getter interface {
    \tGet(key string) string
    }

    type ID = string

    func New() *Store {
    \treturn &Store{}
    }

    func (s *Store) Get(key string) string {
    \treturn fmt.Sprint(s.items[key])
    }

    func Map[T any](xs []T) []T { return xs }
    """
)


def test_go_symbols() -> None:
    symbols = extract_symbols("internal/store/store.go", GO_SOURCE, "go")
    assert kinds(symbols) == {
        ("const", "Version", None),
        ("struct", "Store", None),
        ("interface", "Getter", None),
        ("type", "ID", None),
        ("function", "New", None),
        ("method", "Get", "Store"),
        ("function", "Map", None),
    }
    named = by_name(symbols)
    assert (named["Store"].line, named["Store"].end_line) == (10, 12)
    assert (named["Get"].line, named["Get"].end_line) == (24, 26)


RUST_SOURCE = textwrap.dedent(
    """\
    use std::collections::HashMap;
    pub struct Point { x: i32, y: i32 }
    pub enum Shape { Circle, Square }
    pub trait Draw {
        fn draw(&self);
    }
    pub type Res<T> = Result<T, String>;
    const MAX: usize = 10;
    impl Point {
        pub fn new(x: i32, y: i32) -> Self {
            Point { x, y }
        }
        fn dist<'a>(&'a self, other: &'a Point) -> f64 { let _c = '}'; 0.0 }
    }
    impl<T: Clone> Draw for Wrapper<T> {
        fn draw(&self) {}
    }
    pub async fn run() -> Res<()> { Ok(()) }
    #[cfg(test)]
    mod tests {
        #[test]
        fn it_works() {}
    }
    """
)


def test_rust_symbols() -> None:
    symbols = extract_symbols("src/lib.rs", RUST_SOURCE, "rust")
    assert kinds(symbols) == {
        ("struct", "Point", None),
        ("enum", "Shape", None),
        ("interface", "Draw", None),
        ("method", "draw", "Draw"),
        ("type", "Res", None),
        ("const", "MAX", None),
        ("method", "new", "Point"),
        ("method", "dist", "Point"),
        ("method", "draw", "Wrapper"),
        ("function", "run", None),
        ("function", "it_works", "tests"),
    }


JAVA_SOURCE = textwrap.dedent(
    """\
    package com.example.app;

    import java.util.List;

    @Service
    public class UserService extends Base implements Api {
        private final Repo repo;

        public UserService(Repo repo) {
            this.repo = repo;
        }

        @Override
        public List<User> findAll(int limit) throws IOException {
            if (limit > 0) {
                return repo.find(limit);
            }
            return List.of();
        }

        public interface Listener {
            void onEvent(Event e);
        }

        enum Kind { A, B }
    }
    """
)


def test_java_symbols() -> None:
    symbols = extract_symbols("src/main/java/com/example/app/UserService.java", JAVA_SOURCE, "java")
    assert kinds(symbols) == {
        ("class", "UserService", None),
        ("method", "UserService", "UserService"),
        ("method", "findAll", "UserService"),
        ("interface", "Listener", "UserService"),
        ("method", "onEvent", "Listener"),
        ("enum", "Kind", "UserService"),
    }
    named = by_name(symbols)
    assert named["findAll"].line == 14
    assert named["findAll"].end_line == 19


def test_kotlin_and_csharp_symbols() -> None:
    kotlin = textwrap.dedent(
        """\
        data class User(val name: String)
        class Service(private val repo: Repo) {
            fun save(u: User) {
                println(u)
            }
        }
        fun main() {}
        """
    )
    assert kinds(extract_symbols("a.kt", kotlin, "kotlin")) == {
        ("class", "User", None),
        ("class", "Service", None),
        ("method", "save", "Service"),
        ("function", "main", None),
    }
    csharp = textwrap.dedent(
        """\
        namespace App {
            public class OrderService {
                public OrderService(IRepo repo) { }
                public async Task<Order> GetAsync(int id) {
                    var x = Compute(id);
                    return x;
                }
            }
        }
        """
    )
    assert kinds(extract_symbols("a.cs", csharp, "csharp")) == {
        ("class", "OrderService", None),
        ("method", "OrderService", "OrderService"),
        ("method", "GetAsync", "OrderService"),
    }


def test_unknown_language_and_empty_text() -> None:
    assert extract_symbols("x.txt", "anything", None) == []
    assert extract_symbols("x.py", "", "python") == []
    assert extract_imports("x.txt", "import x", None) == []


# --------------------------------------------------------------------------- imports


def test_python_imports_absolute_and_relative() -> None:
    source = textwrap.dedent(
        """\
        import os, sys as system
        import pkg.sub.mod
        from pkg.core import Engine
        from . import util
        from .helpers import fmt, parse as p
        from ..base import Base
        from .star import *

        def lazy():
            import json
        """
    )
    assert extract_imports("pkg/sub/a.py", source, "python") == [
        "os",
        "sys",
        "pkg.sub.mod",
        "pkg.core",
        "pkg.core.Engine",
        "pkg.sub",
        "pkg.sub.util",
        "pkg.sub.helpers",
        "pkg.sub.helpers.fmt",
        "pkg.sub.helpers.parse",
        "pkg.base",
        "pkg.base.Base",
        "pkg.sub.star",
        "json",
    ]


def test_python_relative_import_spec_example_and_src_layout() -> None:
    assert extract_imports("pkg/a.py", "from .x import y\n", "python") == ["pkg.x", "pkg.x.y"]
    # ``src/`` is a source root, so it is not part of the module name
    assert extract_imports("src/pkg/a.py", "from .x import y\n", "python") == ["pkg.x", "pkg.x.y"]
    # ``__init__.py``: ``.`` is the package itself
    assert extract_imports("pkg/__init__.py", "from .core import run\n", "python") == ["pkg.core", "pkg.core.run"]
    # relative import escaping the top level is dropped; top-level siblings are kept
    assert extract_imports("a.py", "from ..x import y\n", "python") == []
    assert extract_imports("a.py", "from . import sibling\n", "python") == ["sibling"]


def test_python_imports_regex_fallback() -> None:
    source = "from .x import y, z\nimport a.b\ndef broken(:\n"
    assert extract_imports("pkg/m.py", source, "python") == ["pkg.x", "pkg.x.y", "pkg.x.z", "a.b"]


def test_js_imports() -> None:
    source = textwrap.dedent(
        """\
        import React from 'react';
        import type { T } from "../types";
        import * as fs from 'node:fs';
        import def, {
          a,
          b,
        } from './multi';
        import './side-effect.css';
        export * from './reexport';
        export { x as y } from "./named";
        const lazy = await import('./lazy');
        const legacy = require("./legacy");
        """
    )
    assert extract_imports("src/app.ts", source, "typescript") == [
        "react",
        "../types",
        "node:fs",
        "./multi",
        "./side-effect.css",
        "./reexport",
        "./named",
        "./lazy",
        "./legacy",
    ]


def test_go_imports() -> None:
    assert extract_imports("internal/store/store.go", GO_SOURCE, "go") == ["fmt", "example.com/demo/internal/util"]
    assert extract_imports("main.go", 'package main\nimport "os"\n', "go") == ["os"]


def test_rust_imports() -> None:
    source = textwrap.dedent(
        """\
        use std::collections::HashMap;
        use crate::config::{Config, load as load_config};
        pub use super::util::*;
        mod parser;
        pub(crate) mod lexer;
        """
    )
    assert extract_imports("src/lib.rs", source, "rust") == [
        "std::collections::HashMap",
        "crate::config",
        "crate::config::Config",
        "crate::config::load",
        "super::util",
        "self::parser",
        "self::lexer",
    ]


def test_java_kotlin_csharp_c_imports() -> None:
    assert extract_imports("A.java", JAVA_SOURCE, "java") == ["java.util.List"]
    java = "import static org.junit.Assert.assertEquals;\nimport com.x.util.*;\n"
    assert extract_imports("B.java", java, "java") == ["org.junit.Assert.assertEquals", "com.x.util.*"]
    assert extract_imports("a.kt", "import kotlinx.coroutines.launch\n", "kotlin") == ["kotlinx.coroutines.launch"]
    assert extract_imports("a.cs", "using System;\nusing IO = System.IO;\n", "csharp") == ["System", "System.IO"]
    assert extract_imports("a.c", '#include <stdio.h>\n#include "util/str.h"\n', "c") == ["util/str.h"]
