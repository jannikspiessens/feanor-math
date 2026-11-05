use std::fmt::Debug;

use super::*;
use crate::delegate::DelegateRing;
use crate::impl_eq_based_self_iso;
use crate::integer::{BigIntRing, BigIntRingBase, int_cast};
use crate::pid::PrincipalIdealRing;
use crate::primitive_int::StaticRing;
use crate::specialization::*;
use crate::{impl_field_wrap_unwrap_homs, impl_field_wrap_unwrap_isos};

/// `a + b * c + carry` as `(low, high)` 64-bit words; cannot overflow, since
/// `(2^64 - 1) + (2^64 - 1)^2 + (2^64 - 1) = 2^128 - 1`.
#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + (b as u128) * (c as u128) + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// `a + b` as `(low, high)` 64-bit words.
#[inline(always)]
fn adc(a: u64, b: u64) -> (u64, u64) {
    let t = a as u128 + b as u128;
    (t as u64, (t >> 64) as u64)
}

fn u128_to_bigint(x: u128) -> El<BigIntRing> {
    let ZZ = BigIntRing::RING;
    let mut hi = int_cast((x >> 64) as i128, ZZ, StaticRing::<i128>::RING);
    ZZ.mul_pow_2(&mut hi, 64);
    ZZ.add(hi, int_cast((x as u64) as i128, ZZ, StaticRing::<i128>::RING))
}

fn bigint_to_u128(x: &El<BigIntRing>) -> u128 {
    let ZZ = BigIntRing::RING;
    assert!(!ZZ.is_neg(x) && ZZ.abs_highest_set_bit(x).map_or(true, |b| b < 128));
    let mut hi = ZZ.clone_el(x);
    ZZ.euclidean_div_pow_2(&mut hi, 64);
    let mut hi_shifted = ZZ.clone_el(&hi);
    ZZ.mul_pow_2(&mut hi_shifted, 64);
    let lo = ZZ.sub_ref_fst(x, hi_shifted);
    let hi = int_cast(hi, StaticRing::<i128>::RING, ZZ) as u128;
    let lo = int_cast(lo, StaticRing::<i128>::RING, ZZ) as u128;
    (hi << 64) | lo
}

/// Implementation of `Z/nZ` for odd `n` with `2 < n < 2^128`, using Montgomery multiplication
/// with `R = 2^128` on two 64-bit limbs.
///
/// Elements are stored in Montgomery form `x * 2^128 mod n`, always fully reduced to
/// `{ 0, ..., n - 1 }`. Hence equality and zero checks are comparisons, and additions only need
/// to check for a carry (since `2n` may exceed `2^128`).
///
/// The associated integer ring is [`BigIntRing`], so lifts and the image of integers are
/// computed with arbitrary-precision integers; they are not meant to be used in hot loops.
/// In particular, [`FiniteRing::random_element()`] and [`ZnRing::smallest_positive_lift()`]
/// agree with those of [`zn_big::Zn`] over [`BigIntRing`] with the same modulus.
///
/// # Example
/// ```rust
/// # use feanor_math::ring::*;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_128::*;
/// # use feanor_math::assert_el_eq;
/// let ring = Zn::new((1u128 << 127) - 1);
/// let x = ring.int_hom().map(-1);
/// assert_el_eq!(ring, ring.one(), ring.mul_ref(&x, &x));
/// ```
#[derive(Clone)]
pub struct ZnBase {
    modulus: u128,
    modulus_bigint: El<BigIntRing>,
    /// `-n^-1 mod 2^64`
    neg_inv_modulus: u64,
    /// `2^128 mod n`, i.e. the Montgomery form of `1`
    one: u128,
    /// `2^256 mod n`, for converting into Montgomery form
    r2: u128,
    /// `2^384 mod n`, for computing inverses in Montgomery form
    r3: u128,
}

/// An element of [`ZnBase`], in Montgomery form.
#[derive(Clone, Copy, Debug)]
pub struct ZnEl(u128);

pub type Zn = RingValue<ZnBase>;

impl Zn {
    pub fn new(modulus: u128) -> Self { RingValue::from(ZnBase::new(modulus)) }

    /// Creates `Z/nZ` if the modulus `n` is supported, i.e. odd with `2 < n < 2^128`, and
    /// returns `None` otherwise.
    pub fn try_from_bigint(modulus: &El<BigIntRing>) -> Option<Self> {
        let ZZ = BigIntRing::RING;
        let supported = !ZZ.is_neg(modulus)
            && ZZ.abs_highest_set_bit(modulus).is_some_and(|b| b < 128)
            && !ZZ.is_even(modulus)
            && ZZ.is_gt(modulus, &ZZ.int_hom().map(2));
        supported.then(|| Self::new(bigint_to_u128(modulus)))
    }
}

impl ZnBase {
    pub fn new(modulus: u128) -> Self {
        assert!(modulus > 2, "the modulus must be > 2");
        assert!(modulus % 2 == 1, "Montgomery multiplication requires an odd modulus");
        let n0 = modulus as u64;
        // Newton iteration for n0^-1 mod 2^64; each step doubles the number of correct bits,
        // starting with 3 correct bits since n0 * n0 = 1 mod 8
        let mut inv = n0;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
        }
        debug_assert_eq!(n0.wrapping_mul(inv), 1);
        let one = 0u128.wrapping_sub(modulus) % modulus;
        let mut result = ZnBase {
            modulus,
            modulus_bigint: u128_to_bigint(modulus),
            neg_inv_modulus: inv.wrapping_neg(),
            one,
            r2: 0,
            r3: 0,
        };
        // 2^256 mod n by doubling 2^128 mod n another 128 times
        let mut r2 = one;
        for _ in 0..128 {
            r2 = result.add_mod(r2, r2);
        }
        result.r2 = r2;
        result.r3 = result.mont_mul(r2, r2);
        return result;
    }

    pub fn modulus_u128(&self) -> u128 { self.modulus }

    /// `a + b mod n` for `a, b < n`
    #[inline(always)]
    fn add_mod(&self, a: u128, b: u128) -> u128 {
        let (s, carry) = a.overflowing_add(b);
        if carry || s >= self.modulus { s.wrapping_sub(self.modulus) } else { s }
    }

    /// `a - b mod n` for `a, b < n`
    #[inline(always)]
    fn sub_mod(&self, a: u128, b: u128) -> u128 {
        let (d, borrow) = a.overflowing_sub(b);
        if borrow { d.wrapping_add(self.modulus) } else { d }
    }

    /// Montgomery product `a * b * 2^-128 mod n` for `a, b < n`, computed with the coarsely
    /// integrated operand scanning (CIOS) method on two 64-bit limbs.
    #[inline(always)]
    fn mont_mul(&self, a: u128, b: u128) -> u128 {
        let (a0, a1) = (a as u64, (a >> 64) as u64);
        let (b0, b1) = (b as u64, (b >> 64) as u64);
        let (n0, n1) = (self.modulus as u64, (self.modulus >> 64) as u64);

        // t = a * b0
        let (t0, c) = mac(0, a0, b0, 0);
        let (t1, t2) = mac(0, a1, b0, c);
        // t = (t + m * n) / 2^64
        let m = t0.wrapping_mul(self.neg_inv_modulus);
        let (_, c) = mac(t0, m, n0, 0);
        let (t0, c) = mac(t1, m, n1, c);
        let (t1, t2) = adc(t2, c);

        // t = t + a * b1
        let (t0, c) = mac(t0, a0, b1, 0);
        let (t1, c) = mac(t1, a1, b1, c);
        let (t2, t3) = adc(t2, c);
        // t = (t + m * n) / 2^64
        let m = t0.wrapping_mul(self.neg_inv_modulus);
        let (_, c) = mac(t0, m, n0, 0);
        let (t0, c) = mac(t1, m, n1, c);
        let (t1, c) = adc(t2, c);
        let t2 = t3 + c;

        // now t < 2n, where t2 is the bit 2^128
        let result = ((t1 as u128) << 64) | t0 as u128;
        if t2 != 0 || result >= self.modulus { result.wrapping_sub(self.modulus) } else { result }
    }

    #[inline(always)]
    fn to_montgomery(&self, x: u128) -> u128 {
        debug_assert!(x < self.modulus);
        self.mont_mul(x, self.r2)
    }

    #[inline(always)]
    fn from_montgomery(&self, x: u128) -> u128 { self.mont_mul(x, 1) }

    /// The element `x mod n`, for `x < n`.
    pub fn from_u128_promise_reduced(&self, x: u128) -> ZnEl { ZnEl(self.to_montgomery(x)) }

    /// `(a + n) / 2` for odd `a < n`, without overflow
    #[inline(always)]
    fn half_mod(&self, a: u128) -> u128 {
        if a % 2 == 0 { a >> 1 } else { (a >> 1) + (self.modulus >> 1) + 1 }
    }

    /// The inverse of `a` modulo `n` (not in Montgomery form), using the binary extended
    /// Euclidean algorithm, or `None` if `gcd(a, n) != 1`.
    fn inverse_mod(&self, a: u128) -> Option<u128> {
        if a == 0 {
            return None;
        }
        // invariants: x1 * a = u mod n and x2 * a = v mod n
        let (mut u, mut v) = (a, self.modulus);
        let (mut x1, mut x2) = (1u128, 0u128);
        while u != 1 && v != 1 {
            while u % 2 == 0 {
                u >>= 1;
                x1 = self.half_mod(x1);
            }
            while v % 2 == 0 {
                v >>= 1;
                x2 = self.half_mod(x2);
            }
            if u >= v {
                u -= v;
                x1 = self.sub_mod(x1, x2);
            } else {
                v -= u;
                x2 = self.sub_mod(x2, x1);
            }
            // u = v happens only if gcd(a, n) = u > 1
            if u == 0 || v == 0 {
                return None;
            }
        }
        Some(if u == 1 { x1 } else { x2 })
    }

    /// The inverse of `el`, or `None` if it is not a unit.
    pub fn invert_unit(&self, el: &ZnEl) -> Option<ZnEl> {
        // el.0 = a * R, so its inverse is a^-1 * R^-1; multiplying by R^3 in Montgomery
        // arithmetic gives a^-1 * R^-1 * R^3 * R^-1 = a^-1 * R
        self.inverse_mod(el.0).map(|inv| ZnEl(self.mont_mul(inv, self.r3)))
    }
}

impl PartialEq for ZnBase {
    fn eq(&self, other: &Self) -> bool { self.modulus == other.modulus }
}

impl Debug for ZnBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Z/{}Z", self.modulus) }
}

impl RingBase for ZnBase {
    type Element = ZnEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element { *val }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { lhs.0 = self.add_mod(lhs.0, rhs.0); }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.add_assign(lhs, *rhs) }

    fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { lhs.0 = self.sub_mod(lhs.0, rhs.0); }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.sub_assign(lhs, *rhs) }

    fn negate_inplace(&self, lhs: &mut Self::Element) { lhs.0 = self.sub_mod(0, lhs.0); }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) { lhs.0 = self.mont_mul(lhs.0, rhs.0); }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) { self.mul_assign(lhs, *rhs) }

    fn square(&self, value: &mut Self::Element) { value.0 = self.mont_mul(value.0, value.0); }

    fn fma(&self, lhs: &Self::Element, rhs: &Self::Element, summand: Self::Element) -> Self::Element {
        ZnEl(self.add_mod(self.mont_mul(lhs.0, rhs.0), summand.0))
    }

    fn zero(&self) -> Self::Element { ZnEl(0) }

    fn one(&self) -> Self::Element { ZnEl(self.one) }

    fn neg_one(&self) -> Self::Element { ZnEl(self.sub_mod(0, self.one)) }

    fn from_int(&self, value: i32) -> Self::Element {
        let abs = ZnEl(self.to_montgomery(value.unsigned_abs() as u128 % self.modulus));
        if value < 0 { self.negate(abs) } else { abs }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool { lhs.0 == rhs.0 }

    fn is_zero(&self, value: &Self::Element) -> bool { value.0 == 0 }

    fn is_one(&self, value: &Self::Element) -> bool { value.0 == self.one }

    fn is_commutative(&self) -> bool { true }

    fn is_noetherian(&self) -> bool { true }

    fn is_approximate(&self) -> bool { false }

    fn dbg_within<'a>(
        &self,
        value: &Self::Element,
        out: &mut std::fmt::Formatter<'a>,
        _: EnvBindingStrength,
    ) -> std::fmt::Result {
        write!(out, "{}", self.from_montgomery(value.0))
    }

    fn characteristic<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
    where
        I::Type: IntegerRing,
    {
        self.size(ZZ)
    }
}

impl_eq_based_self_iso! { ZnBase }

impl DivisibilityRing for ZnBase {
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(lhs) {
            return Some(self.zero());
        }
        if let Some(inv) = self.invert_unit(rhs) {
            Some(self.mul(*lhs, inv))
        } else {
            // zero divisors only exist for composite moduli; use the generic implementation
            super::generic_impls::checked_left_div(RingRef::new(self), lhs, rhs)
        }
    }

    fn invert(&self, el: &Self::Element) -> Option<Self::Element> { self.invert_unit(el) }
}

impl PrincipalIdealRing for ZnBase {
    fn checked_div_min(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        super::generic_impls::checked_div_min(RingRef::new(self), lhs, rhs)
    }

    fn extended_ideal_gen(
        &self,
        lhs: &Self::Element,
        rhs: &Self::Element,
    ) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = self
            .integer_ring()
            .extended_ideal_gen(&self.smallest_positive_lift(*lhs), &self.smallest_positive_lift(*rhs));
        let quo = RingRef::new(self).into_can_hom(self.integer_ring()).ok().unwrap();
        (quo.map(s), quo.map(t), quo.map(d))
    }
}

impl<J: IntegerRing + ?Sized> CanHomFrom<J> for ZnBase
where
    J: CanIsoFromTo<BigIntRingBase>,
{
    type Homomorphism = super::generic_impls::BigIntToZnHom<J, BigIntRingBase, ZnBase>;

    fn has_canonical_hom(&self, from: &J) -> Option<Self::Homomorphism> {
        super::generic_impls::has_canonical_hom_from_bigint(from, self, BigIntRing::RING.get_ring(), None)
    }

    fn map_in(&self, from: &J, el: J::Element, hom: &Self::Homomorphism) -> Self::Element {
        super::generic_impls::map_in_from_bigint(
            from,
            self,
            BigIntRing::RING.get_ring(),
            el,
            hom,
            |n| self.from_int_promise_reduced(n),
            |n| self.from_int_promise_reduced(BigIntRing::RING.euclidean_rem(n, &self.modulus_bigint)),
        )
    }
}

#[derive(Clone, Copy)]
pub struct ZnBaseElementsIter<'a> {
    ring: &'a ZnBase,
    current: u128,
}

impl<'a> Iterator for ZnBaseElementsIter<'a> {
    type Item = ZnEl;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.modulus {
            let result = self.ring.from_u128_promise_reduced(self.current);
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }
}

impl FiniteRingSpecializable for ZnBase {
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> O::Output { op.execute() }
}

impl FiniteRing for ZnBase {
    type ElementsIter<'a> = ZnBaseElementsIter<'a>;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> { ZnBaseElementsIter { ring: self, current: 0 } }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        super::generic_impls::random_element(self, rng)
    }

    fn size<I: RingStore + Copy>(&self, other_ZZ: I) -> Option<El<I>>
    where
        I::Type: IntegerRing,
    {
        if other_ZZ.get_ring().representable_bits().is_none()
            || self.integer_ring().abs_log2_ceil(&self.integer_ring().add_ref_fst(&self.modulus_bigint, self.integer_ring().one()))
                <= other_ZZ.get_ring().representable_bits()
        {
            Some(int_cast(self.integer_ring().clone_el(&self.modulus_bigint), other_ZZ, self.integer_ring()))
        } else {
            None
        }
    }
}

impl ZnRing for ZnBase {
    type IntegerRingBase = BigIntRingBase;
    type IntegerRing = BigIntRing;

    fn integer_ring(&self) -> &Self::IntegerRing { &BigIntRing::RING }

    fn modulus(&self) -> &El<Self::IntegerRing> { &self.modulus_bigint }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::IntegerRing> {
        u128_to_bigint(self.from_montgomery(el.0))
    }

    fn from_int_promise_reduced(&self, x: El<Self::IntegerRing>) -> Self::Element {
        self.from_u128_promise_reduced(bigint_to_u128(&x))
    }
}

impl_field_wrap_unwrap_homs! { ZnBase, ZnBase }
impl_field_wrap_unwrap_isos! { ZnBase, ZnBase }

#[cfg(test)]
const TEST_MODULI: [u128; 8] = [
    3,
    17,
    65537,
    (1 << 64) + 13,
    (1 << 127) - 1,
    // 248^16 + 1, the GBFV plaintext modulus that motivated this ring; > 2^127, so 2n > 2^128
    204751406252581656212043048442748993537,
    u128::MAX - 158, // the largest prime below 2^128
    u128::MAX,       // 2^128 - 1, composite
];

#[cfg(test)]
fn test_elements(ring: &Zn) -> Vec<ZnEl> {
    let n = ring.get_ring().modulus_u128();
    let mut rng = oorandom::Rand64::new(1);
    let mut values: Vec<u128> = [0, 1, 2, n / 2, n - 2, n - 1].into_iter().map(|x| x % n).collect();
    values.extend((0..10).map(|_| ring.get_ring().from_montgomery(ring.random_element(|| rng.rand_u64()).0)));
    // the field axiom test requires distinct elements
    values.sort();
    values.dedup();
    values.into_iter().map(|x| ring.get_ring().from_u128_promise_reduced(x)).collect()
}

#[test]
fn test_mul_against_bigint() {
    let ZZ = BigIntRing::RING;
    let mut rng = oorandom::Rand64::new(2);
    for n in TEST_MODULI {
        let ring = Zn::new(n);
        let ring_ref = zn_big::Zn::new(ZZ, u128_to_bigint(n));
        let to_ref = |x: ZnEl| ring_ref.coerce(&ZZ, ring.smallest_positive_lift(x));
        for _ in 0..1000 {
            let x = ring.random_element(|| rng.rand_u64());
            let y = ring.random_element(|| rng.rand_u64());
            assert_el_eq!(ring_ref, ring_ref.mul(to_ref(x), to_ref(y)), to_ref(ring.mul(x, y)));
            assert_el_eq!(ring_ref, ring_ref.add(to_ref(x), to_ref(y)), to_ref(ring.add(x, y)));
            assert_el_eq!(ring_ref, ring_ref.sub(to_ref(x), to_ref(y)), to_ref(ring.sub(x, y)));
        }
    }
}

#[test]
fn test_random_element_agrees_with_zn_big() {
    let ZZ = BigIntRing::RING;
    for n in TEST_MODULI {
        let ring = Zn::new(n);
        let ring_ref = zn_big::Zn::new(ZZ, u128_to_bigint(n));
        let mut rng1 = oorandom::Rand64::new(3);
        let mut rng2 = oorandom::Rand64::new(3);
        for _ in 0..100 {
            let x = ring.smallest_positive_lift(ring.random_element(|| rng1.rand_u64()));
            let y = ring_ref.smallest_positive_lift(ring_ref.random_element(|| rng2.rand_u64()));
            assert_el_eq!(ZZ, y, x);
        }
    }
}

#[test]
fn test_try_from_bigint() {
    let ZZ = BigIntRing::RING;
    for n in TEST_MODULI {
        assert!(Zn::try_from_bigint(&u128_to_bigint(n)).unwrap().get_ring().modulus_u128() == n);
    }
    assert!(Zn::try_from_bigint(&ZZ.power_of_two(128)).is_none());
    assert!(Zn::try_from_bigint(&ZZ.add(ZZ.power_of_two(128), ZZ.one())).is_none());
    assert!(Zn::try_from_bigint(&ZZ.int_hom().map(1 << 20)).is_none());
    assert!(Zn::try_from_bigint(&ZZ.int_hom().map(1)).is_none());
}

#[test]
fn test_invert() {
    for n in TEST_MODULI {
        let ring = Zn::new(n);
        let is_prime = ring.is_field();
        for x in test_elements(&ring) {
            match ring.invert(&x) {
                Some(inv) => assert_el_eq!(ring, ring.one(), ring.mul(x, inv)),
                None => assert!(ring.is_zero(&x) || !is_prime),
            }
        }
    }
    // 2^128 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 274177 * 6700417 * 67280421310721
    let ring = Zn::new(u128::MAX);
    assert!(ring.invert(&ring.int_hom().map(3)).is_none());
    assert!(ring.invert(&ring.int_hom().map(7)).is_some());
}

#[test]
fn test_ring_axioms() {
    for n in TEST_MODULI {
        let ring = Zn::new(n);
        crate::ring::generic_tests::test_ring_axioms(&ring, test_elements(&ring).into_iter());
    }
}

#[test]
fn test_divisibility_axioms() {
    for n in TEST_MODULI {
        let ring = Zn::new(n);
        crate::divisibility::generic_tests::test_divisibility_axioms(&ring, test_elements(&ring).into_iter());
    }
    // a composite modulus, so that there are zero divisors
    let ring = Zn::new(63);
    crate::divisibility::generic_tests::test_divisibility_axioms(&ring, ring.elements());
}

#[test]
fn test_principal_ideal_ring_axioms() {
    for n in [3, 17, 63] {
        let ring = Zn::new(n);
        crate::pid::generic_tests::test_principal_ideal_ring_axioms(&ring, ring.elements());
    }
}

#[test]
fn test_zn_axioms() {
    for n in [3, 17, 63, 255, 257] {
        super::generic_tests::test_zn_axioms(Zn::new(n));
    }
}

#[test]
fn test_map_in_large_int() {
    for n in TEST_MODULI {
        super::generic_tests::test_map_in_large_int(Zn::new(n));
    }
}

#[test]
fn test_finite_ring_axioms() {
    for n in [3, 17, 63, 255, 257] {
        crate::rings::finite::generic_tests::test_finite_ring_axioms(&Zn::new(n));
    }
}

#[test]
fn test_field_axioms() {
    for n in [3, 17, 65537, (1 << 127) - 1, 204751406252581656212043048442748993537, u128::MAX - 158] {
        let field = Zn::new(n).as_field().ok().unwrap();
        let els = test_elements(&Zn::new(n));
        crate::field::generic_tests::test_field_axioms(&field, els.into_iter().map(|x| field.get_ring().rev_delegate(x)));
    }
}
