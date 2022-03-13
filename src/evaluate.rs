pub mod and {
    use crate::mult_triple::MultTriple;

    pub fn compute_shares(x: bool, y: bool, mt: &MultTriple) -> (bool, bool) {
        let d = x ^ mt.get_a();
        let e = y ^ mt.get_b();
        (d, e)
    }

    pub fn evaluate(d: [bool; 2], e: [bool; 2], mt: MultTriple, party_id: usize) -> bool {
        let d = d[0] ^ d[1];
        let e = e[0] ^ e[1];
        let res = d & mt.get_b() ^ e & mt.get_a() ^ mt.get_c();
        if party_id == 0 {
            res ^ d & e
        } else {
            res
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::evaluate::and;
    use crate::mult_triple::MultTriple;

    #[test]
    fn and_eval() {
        let p0 = [false, true];
        let p1 = [true, false];
        let mt0 = MultTriple::zeroes();
        let mt1 = MultTriple::zeroes();
        let shares_0 = and::compute_shares(p0[0], p0[1], &mt0);
        let shares_1 = and::compute_shares(p1[0], p1[1], &mt1);
        let a0 = and::evaluate([shares_0.0, shares_1.0], [shares_0.1, shares_1.1], mt0, 0);
        let a1 = and::evaluate([shares_0.0, shares_1.0], [shares_0.1, shares_1.1], mt1, 1);
        assert_eq!(true, a0 ^ a1);
    }
}
