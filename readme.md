# Kompilieren und Ausführen

Das Projekt benötigt die Nightly toolchain, getestet habe ich zuletzt `1.73.0-nightly  (0e8e857b1 2023-07-16)`.
Die Nightly toolchain kann mit dem folgenden Befehl installiert werden:

```bash
rustup toolchain install nightly-2023-07-16
```

Das Frontend ist in einem Example implementiert.
Zum kompilieren muss die Environment Variable `RUSTFLAGS` gesetzt werden:
```bash
export RUSTFLAGS="-C target-cpu=native"
cargo +nightly run --release --example ads_project -- <args>
```

Um das Kompilieren und Ausführen einfacher zu machen, gibt es das `run.sh` script.
Es nimmt die Argumente gemäß der Aufgabenstellung entgegen und kompiliert und führt das Programm aus.

```bash
run.sh rmq input_rmq_1.txt output_rmq_1.txt
```

Es werden ca 200 dependencies mitkompiliert; keine davon wird gegen das Projekt gelinkt.
Leider kann man in Rust dev-dependencies nicht optional deklarieren.