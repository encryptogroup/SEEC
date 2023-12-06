use nom::character::complete::{digit1, multispace0};
use nom::combinator::map_res;
use nom::error::{ErrorKind, FromExternalError, ParseError};
use nom::sequence::delimited;
use nom::{IResult, Parser};
use smallvec::SmallVec;
use std::num::ParseIntError;

pub mod bristol;
pub mod fuse;

fn integer<'a, E: ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + 'a>(
    i: &'a str,
) -> IResult<&'a str, usize, E> {
    map_res(digit1, |s: &str| s.parse())(i)
}

fn integer_ws<'a, E: ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + 'a>(
    i: &'a str,
) -> IResult<&'a str, usize, E> {
    ws(integer)(i)
}

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading and
/// trailing whitespace, returning the output of `inner`.
/// Source: https://docs.rs/nom/latest/nom/recipes/index.html#wrapper-combinators-that-eat-whitespace-before-and-after-a-parser
fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

/// count parser from nom adapted to return smallvec
#[allow(unused)]
fn count_sm<I, O, E, F, const N: usize>(
    mut f: F,
    count: usize,
) -> impl FnMut(I) -> IResult<I, SmallVec<[O; N]>, E>
where
    I: Clone + PartialEq,
    F: Parser<I, O, E>,
    E: ParseError<I>,
{
    move |i: I| {
        let mut input = i.clone();
        let mut res = SmallVec::new();

        for _ in 0..count {
            let input_ = input.clone();
            match f.parse(input_) {
                Ok((i, o)) => {
                    res.push(o);
                    input = i;
                }
                Err(nom::Err::Error(e)) => {
                    return Err(nom::Err::Error(E::append(i, ErrorKind::Count, e)));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok((input, res))
    }
}
