#include "SilentOT_Tests.h"

#include "libOTe/Tools/SilentPprf.h"
#include "libOTe/TwoChooseOne/Silent/SilentOtExtSender.h"
#include "libOTe/TwoChooseOne/Silent/SilentOtExtReceiver.h"
#include <cryptoTools/Common/Log.h>
#include <cryptoTools/Common/BitVector.h>
#include <cryptoTools/Common/Range.h>
#include <cryptoTools/Network/IOService.h>
#include <cryptoTools/Common/TestCollection.h>
#include "libOTe/Tools/Tools.h"
#include "libOTe/Tools/QuasiCyclicCode.h"
#include "Common.h"
using namespace oc;

void Tools_bitShift_test(const CLP& cmd)
{
#ifdef ENABLE_BITPOLYMUL
    //u64 nBits = ;
    u64 n = cmd.getOr("n", 10);// (nBits + 127) / 128;
    u64 t = cmd.getOr("t", 10);

    PRNG prng(toBlock(cmd.getOr("seed", 0)));

    std::vector<block> dest(n), in;
    prng.get(dest.data(), dest.size());



    //std::cout << "a " << (_mm_slli_epi64(AllOneBlock, 20)) << std::endl;
    //std::cout << "b " << (_mm_srli_epi64(AllOneBlock, 20)) << std::endl;

    for (u64 i = 0; i < t; ++i)
    {
        u8 bitShift = prng.get<u8>() % 128;

        u64 inSize = std::max<u64>(1, n + (i & 1 ? 1 : -1));
        u64 inBits = std::min<u64>(n * 128, inSize * 128 - bitShift);

        in.resize(inSize);
        prng.get(in.data(), in.size());


        memset(dest.data(), 0, dest.size() * 16);

        BitVector dv((u8*)dest.data(), n * 128);
        BitVector iv, ivt((u8*)in.data(), in.size() * 128);
        iv.append((u8*)in.data(), inBits, bitShift);
        iv.resize(n * 128, 0);
        auto dv1 = dv;

        dv ^= iv;

        QuasiCyclicCode::bitShiftXor(dest, in, bitShift);


        BitVector dv2((u8*)dest.data(), n * 128);

        if (dv != dv2)
        {
            auto b = (bitShift > 64) ? 128 - bitShift : 64 - bitShift;

            std::cout << "\n" << int(bitShift) << "\n";
            std::cout << " i* " << ivt << std::endl;
            std::cout << " i  " << iv << std::endl;
            std::cout << " d  " << dv1 << std::endl;

            std::cout << "   " << std::string(b, ' ') << std::string(bitShift, 'x') << '\n';
            std::cout << "act " << dv2 << std::endl;
            std::cout << "exp " << dv << std::endl;
            std::cout << "    " << (dv2 ^ dv) << std::endl;
            throw RTE_LOC;
        }

    }
#else
    throw UnitTestSkipped("ENABLE_BITPOLYMUL not defined.");
#endif
}

void clearBits(span<block> in, u64 idx)
{
    auto p = (u8*)in.data() + idx / 8;
    auto e = (u8*)in.data() + in.size() * 16;

    if (idx & 7)
    {
        *p++ &= (1 << (idx & 7)) - 1;
    }

    while (p != e)
        *p++ = 0;


    BitVector test((u8*)in.data(), in.size() * 128);
    for (u64 i = idx; i < test.size(); ++i)
        if (test[i])
            throw RTE_LOC;
}


void Tools_modp_test(const CLP& cmd)
{
#ifdef ENABLE_BITPOLYMUL

    PRNG prng(toBlock(cmd.getOr("seed", 0)));

    auto iBits = cmd.getOr("c", 1026ull);
    auto nBits = cmd.getOr("n", 223ull);

    auto n = (nBits + 127) / 128;
    auto c = (iBits + nBits - 1) / nBits;

    std::vector<block> dest(n), in((iBits + 127) / 128);

    for (u64 i = 0; i < dest.size(); ++i)
    {
        u64 p = nBits;// -(prng.get<u64>() % 128);

        prng.get(in.data(), in.size());
        memset(in.data(), -1, in.size() * 16);
        clearBits(in, iBits);

        memset(dest.data(), 0, dest.size() * 16);

        BitVector dv((u8*)in.data(), p);
        BitVector iv;

        //std::cout << "\nin[0] = " << dv << std::endl;

        for (u64 j = 1; j < c; ++j)
        {
            auto rem = std::min<u64>(p, iBits - j * p);
            iv.resize(0);
            iv.append((u8*)in.data(), rem, j * p);

            //std::cout << "in["<<j<<"] = " << iv << std::endl;

            iv.resize(p, 0);
            dv ^= iv;
        }
        //std::cout << "out   = " << dv << std::endl;



        QuasiCyclicCode::modp(dest, in, p);


        BitVector dv2((u8*)dest.data(), p);

        if (dv != dv2)
        {
            //auto b = (bitShift > 64) ? 128 - bitShift : 64 - bitShift;
            auto diff = (dv2 ^ dv);
            std::cout << "\n" << p << "\n";
            //std::cout << "   "
            //    << std::string(b, ' ')
            //    << std::string(bitShift, 'x') << '\n';
            std::cout << "act     " << dv2 << std::endl;
            std::cout << "exp     " << dv << std::endl;
            std::cout << "f       " << diff << std::endl;

            for (u64 i = 0; i < diff.size(); ++i)
                if (diff[i])
                    std::cout << " " << i;
            std::cout << std::endl;

            throw RTE_LOC;
        }

        //std::cout << dv2 << std::endl;

    }

#else
    throw UnitTestSkipped("ENABLE_BITPOLYMUL not defined.");
#endif
}

void Tools_quasiCyclic_test(const oc::CLP& cmd)
{

#ifdef ENABLE_BITPOLYMUL


    QuasiCyclicCode code;
    u64 nn = 1 << 10;
    u64 t = 10;
    auto scaler = 2;
    //auto secParam = 128;

    auto mP = nextPrime(nn);
    auto n = mP * scaler;
    //auto mNumPartitions = getPartitions(scaler, mP, secParam);
    //auto ss = (mP * scaler + mNumPartitions - 1) / mNumPartitions;
    //mSizePer = roundUpTo(ss, 8);
    //mN2 = mSizePer * mNumPartitions;
    //mN = mN2 / scaler;
    //mScaler = scaler;

    AlignedUnVector<block> A(n), B(n), C(n);

    PRNG prng(oc::ZeroBlock);
    code.init(mP);

    for (auto tt : rng(t))
    {
        (void)tt;

        prng.get(A.data(), n);
        prng.get(B.data(), n);

        for (auto i : rng(n))
        {
            C[i] = A[i] ^ B[i];
        }

        code.dualEncode(A);
        code.dualEncode(B);
        code.dualEncode(C);

        for (u64 i : rng(mP))
        {
            if (C[i] != (A[i] ^ B[i]))
                throw RTE_LOC;
        }


    }


    for (auto tt : rng(t))
    {
        (void)tt;

        for (u64 i = 0; i < n; ++i)
        {
            A[i] = oc::zeroAndAllOne[prng.getBit()];
        }

        code.dualEncode(A);
        
        for (u64 i : rng(mP))
        {

            if (A[i] != oc::AllOneBlock && A[i] != oc::ZeroBlock)
            {
                std::cout << i << " " << A[i] << std::endl;
                throw RTE_LOC;
            }
        }
    }



    {
        
        mP = nextPrime(50);
        n = mP * scaler;
        code.init(mP);
        auto mtx = code.getMatrix();
        A.resize(n);
        auto bb = 10;

        prng.get(A.data(), n);
        DenseMtx AA(bb, n);

        for (auto i : rng(n))
        {
            for (u64 j : rng(bb))
                AA(j, i) = *BitIterator((u8*)&A[i], j);
        }

        code.dualEncode(A);
        auto A2 = AA * mtx;

        for (auto i : rng(mP))
        {
            for (u64 j : rng(bb))
                if (A2(j, i) != *BitIterator((u8*)&A[i], j))
                    throw RTE_LOC;
        }

    }
#else
    throw UnitTestSkipped("ENABLE_BITPOLYMUL not defined.");
#endif
}


#ifdef ENABLE_SILENTOT
namespace {

#ifdef ENABLE_SILENTOT

    void fakeBase(u64 n,
        u64 s,
        u64 threads,
        PRNG& prng,
        SilentOtExtReceiver& recver, SilentOtExtSender& sender)
    {
        sender.configure(n, s, threads);
        auto count = sender.silentBaseOtCount();
        std::vector<std::array<block, 2>> msg2(count);
        for (u64 i = 0; i < msg2.size(); ++i)
        {
            msg2[i][0] = prng.get();
            msg2[i][1] = prng.get();
        }
        sender.setSilentBaseOts(msg2);

        // fake base OTs.
        {
            recver.configure(n, s, threads);
            BitVector choices = recver.sampleBaseChoiceBits(prng);
            std::vector<block> msg(choices.size());
            for (u64 i = 0; i < msg.size(); ++i)
                msg[i] = msg2[i][choices[i]];
            recver.setSilentBaseOts(msg);
        }
    }

    void checkRandom(
        span<block> messages, span<std::array<block, 2>>messages2,
        BitVector& choice, u64 n,
        bool verbose)
    {

        if (messages.size() != n)
            throw RTE_LOC;
        if (messages2.size() != n)
            throw RTE_LOC;
        if (choice.size() != n)
            throw RTE_LOC;
        bool passed = true;

        for (u64 i = 0; i < n; ++i)
        {
            block m1 = messages[i];
            block m2a = messages2[i][0];
            block m2b = (messages2[i][1]);
            u8 c = choice[i];


            std::array<bool, 2> eqq{
                eq(m1, m2a),
                eq(m1, m2b)
            };
            if (eqq[c ^ 1] == true)
            {
                passed = false;
                if (verbose)
                    std::cout << Color::Pink;
            }
            if (eqq[0] == false && eqq[1] == false)
            {
                passed = false;
                if (verbose)
                    std::cout << Color::Red;
            }

            if (eqq[c] == false && verbose)
                std::cout << "m" << i << " " << m1 << " != (" << m2a << " " << m2b << ")_" << (int)c << "\n";

        }

        if (passed == false)
            throw RTE_LOC;
    }


    template<typename Choice>
    void checkCorrelated(
        span<block> Ar, span<block> Bs,
        Choice& choice, block delta, u64 n,
        bool verbose,
        ChoiceBitPacking packing)
    {

        if (Ar.size() != n)
            throw RTE_LOC;
        if (Bs.size() != n)
            throw RTE_LOC;
        if (packing == ChoiceBitPacking::False &&
            (u64)choice.size() != n)
            throw RTE_LOC;
        bool passed = true;
        //bool first = true;
        block mask = AllOneBlock ^ OneBlock;


        for (u64 i = 0; i < n; ++i)
        {
            block m1 = Ar[i];
            block m2a = Bs[i];
            block m2b = (Bs[i] ^ delta);
            u8 c, c2;

            if (packing == ChoiceBitPacking::True)
            {
                c = u8((m1 & OneBlock) == OneBlock) & 1;
                m1 = m1 & mask;
                m2a = m2a & mask;
                m2b = m2b & mask;

                if (choice.size())
                {
                    c2 = choice[i];

                    if (c2 != c)
                        throw RTE_LOC;
                }
            }
            else
            {
                c = choice[i];
            }

            std::array<bool, 2> eqq{
                eq(m1, m2a),
                eq(m1, m2b)
            };

            bool good = true;
            if (eqq[c] == false || eqq[c ^ 1] == true)
            {
                good = passed = false;
                //if (verbose)
                std::cout << Color::Pink;
            }
            if (eqq[0] == false && eqq[1] == false)
            {
                good = passed = false;
                //if (verbose)
                std::cout << Color::Red;
            }

            if (!good /*&& first*/)
            {
                //first = false;
                std::cout << i << " m " << mask << std::endl;
                std::cout << "r " << m1 << " " << int(c) << std::endl;
                std::cout << "s " << m2a << " " << m2b << std::endl;
                std::cout << "d " << (m1 ^ m2a) << " " << (m1 ^ m2b) << std::endl;
            }

            std::cout << Color::Default;
        }

        if (passed == false)
            throw RTE_LOC;
    }

#endif
}
#endif
using namespace tests_libOTe;

void OtExt_Silent_random_Test(const CLP& cmd)
{
#ifdef ENABLE_SILENTOT

    
    auto sockets = cp::LocalAsyncSocket::makePair();

    u64 n = cmd.getOr("n", 10000);
    bool verbose = cmd.getOr("v", 0) > 1;
    u64 threads = cmd.getOr("t", 4);
    u64 s = cmd.getOr("s", 2);

    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));


    SilentOtExtSender sender;
    SilentOtExtReceiver recver;
    fakeBase(n, s, threads, prng, recver, sender);
    auto type = OTType::Random;

    std::vector<block> messages2(n);
    BitVector choice(n);
    std::vector<std::array<block, 2>> messages(n);

    auto p0 = sender.silentSend(messages, prng, sockets[0]);
    auto p1 = recver.silentReceive(choice, messages2, prng, sockets[1], type);

    eval(p0, p1);

    checkRandom(messages2, messages, choice, n, verbose);

#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}


void OtExt_Silent_correlated_Test(const CLP& cmd)
{
#ifdef ENABLE_SILENTOT

    
    auto sockets = cp::LocalAsyncSocket::makePair();

    u64 n = cmd.getOr("n", 10000);
    bool verbose = cmd.getOr("v", 0) > 1;
    u64 threads = cmd.getOr("t", 4);
    u64 s = cmd.getOr("s", 2);

    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));


    SilentOtExtSender sender;
    SilentOtExtReceiver recver;
    fakeBase(n, s, threads, prng, recver, sender);

    block delta = prng.get();

    std::vector<block> messages2(n);
    BitVector choice(n);
    std::vector<block> messages(n);
    auto type = OTType::Correlated;

    auto p0 = sender.silentSend(delta, messages, prng, sockets[0]);
    auto p1 = recver.silentReceive(choice, messages2, prng, sockets[1], type);

    eval(p0, p1);

    checkCorrelated(
        messages, messages2, choice, delta,
        n, verbose, ChoiceBitPacking::False);
#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}



void OtExt_Silent_inplace_Test(const CLP& cmd)
{
#ifdef ENABLE_SILENTOT


    
    auto sockets = cp::LocalAsyncSocket::makePair();

    u64 n = cmd.getOr("n", 10000);
    bool verbose = cmd.getOr("v", 0) > 1;
    u64 threads = cmd.getOr("t", 4);
    u64 s = cmd.getOr("s", 2);

    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));


    SilentOtExtSender sender;
    SilentOtExtReceiver recver;

    block delta = prng.get();

    //auto type = OTType::Correlated;

    {
        fakeBase(n, s, threads, prng, recver, sender);
        auto p0 = sender.silentSendInplace(delta, n, prng, sockets[0]);
        auto p1 = recver.silentReceiveInplace(n, prng, sockets[1]);

        eval(p0, p1);


        auto& messages = recver.mA;
        auto& messages2 = sender.mB;
        auto& choice = recver.mC;
        checkCorrelated(messages, messages2, choice, delta,
            n, verbose, ChoiceBitPacking::False);
    }

    {
        fakeBase(n, s, threads, prng, recver, sender);
        auto p0 = sender.silentSendInplace(delta, n, prng, sockets[0]);
        auto p1 = recver.silentReceiveInplace(n, prng, sockets[1], ChoiceBitPacking::True);

        eval(p0, p1);

        auto& messages = recver.mA;
        auto& messages2 = sender.mB;
        auto& choice = recver.mC;
        checkCorrelated(messages, messages2, choice, delta,
            n, verbose, ChoiceBitPacking::True);

    }
#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}

void OtExt_Silent_paramSweep_Test(const oc::CLP& cmd)
{
#ifdef ENABLE_SILENTOT

    
    auto sockets = cp::LocalAsyncSocket::makePair();

    std::vector<u64> nn = cmd.getManyOr<u64>("n",
        { 12, /*134,*/433 , /*4234,*/5466 });

    bool verbose = cmd.getOr("v", 0) > 1;
    u64 threads = cmd.getOr("t", 4);
    u64 s = cmd.getOr("s", 2);

    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));

    SilentOtExtSender sender;
    SilentOtExtReceiver recver;

    block delta = prng.get();
    //auto type = OTType::Correlated;

    for (auto n : nn)
    {
        fakeBase(n, s, threads, prng, recver, sender);

        auto p0 = sender.silentSendInplace(delta, n, prng, sockets[0]);
        auto p1 = recver.silentReceiveInplace(n, prng, sockets[1]);

        eval(p0, p1);

        checkCorrelated(sender.mB, recver.mA, recver.mC, delta,
            n, verbose, ChoiceBitPacking::False);
    }

#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}


void OtExt_Silent_QuasiCyclic_Test(const oc::CLP& cmd)
{
#if defined(ENABLE_SILENTOT) && defined(ENABLE_BITPOLYMUL)


    
    auto sockets = cp::LocalAsyncSocket::makePair();

    std::vector<u64> nn = cmd.getManyOr<u64>("n",
        { /*12, 134, 600,*/ 4234/*, 14366 */ });//

    bool verbose = cmd.getOr("v", 0) > 1;
    u64 threads = cmd.getOr("t", 4);
    u64 s = cmd.getOr("s", 2);


    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));

    SilentOtExtSender sender;
    SilentOtExtReceiver recver;

    sender.mMultType = MultType::QuasiCyclic;
    recver.mMultType = MultType::QuasiCyclic;

    //sender.mDebug = true;
    //recver.mDebug = true;

    block delta = prng.get();
    //auto type = OTType::Correlated;

    for (auto n : nn)
    {
        //block delta
        std::vector<std::array<block, 2>> msg2(n);
        std::vector<block> msg1(n);
        BitVector choice(n);

        fakeBase(n, s, threads, prng, recver, sender);
        auto p0 = sender.silentSend(msg2, prng, sockets[0]);
        auto p1 = recver.silentReceive(choice, msg1, prng, sockets[1]);

        eval(p0, p1);


        checkRandom(msg1, msg2, choice, n, verbose);

        auto type = ChoiceBitPacking::False;
        fakeBase(n, s, threads, prng, recver, sender);
        p0 = sender.silentSendInplace(delta, n, prng, sockets[0]);
        p1 = recver.silentReceiveInplace(n, prng, sockets[1], type);


        eval(p0, p1);

        checkCorrelated(recver.mA, sender.mB, recver.mC, delta, n, verbose, type);

    }

#else
    throw UnitTestSkipped("ENABLE_SILENTOT or ENABLE_BITPOLYMUL are not defined.");
#endif
}

void OtExt_Silent_Silver_Test(const oc::CLP& cmd)
{

#if defined(ENABLE_SILENTOT) && defined(ENABLE_INSECURE_SILVER)

    auto sockets = cp::LocalAsyncSocket::makePair();

    std::vector<u64> nn = cmd.getManyOr<u64>("n",
        { /*12, 134, 600,*/ 1234/*, 14366 */ });//

    bool verbose = cmd.getOr("v", 0) > 1;
    u64 threads = cmd.getOr("t", 4);
    u64 s = cmd.getOr("s", 2);


    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));

    SilentOtExtSender sender;
    SilentOtExtReceiver recver;

    sender.mMultType = MultType::slv5;
    recver.mMultType = MultType::slv5;

    //sender.mDebug = true;
    //recver.mDebug = true;

    block delta = prng.get();
    //auto type = OTType::Correlated;

    for (auto n : nn)
    {
        //block delta
        std::vector<std::array<block, 2>> msg2(n);
        std::vector<block> msg1(n);
        BitVector choice(n);

        fakeBase(n, s, threads, prng, recver, sender);
        auto p0 = sender.silentSend(msg2, prng, sockets[0]);
        auto p1 = recver.silentReceive(choice, msg1, prng, sockets[1]);

        eval(p0, p1);


        checkRandom(msg1, msg2, choice, n, verbose);

        auto type = ChoiceBitPacking::False;
        fakeBase(n, s, threads, prng, recver, sender);
        p0 = sender.silentSendInplace(delta, n, prng, sockets[0]);
        p1 = recver.silentReceiveInplace(n, prng, sockets[1], type);


        eval(p0, p1);

        checkCorrelated(recver.mA, sender.mB, recver.mC, delta, n, verbose, type);

    }

#else
    throw UnitTestSkipped("ENABLE_SILENTOT or ENABLE_INSECURE_SILVER are not defined.");
#endif
}

void OtExt_Silent_baseOT_Test(const oc::CLP& cmd)
{

#ifdef ENABLE_SILENTOT


    
    auto sockets = cp::LocalAsyncSocket::makePair();

    u64 n = 123;//

    bool verbose = cmd.getOr("v", 0) > 1;

    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));

    SilentOtExtSender sender;
    SilentOtExtReceiver recver;

    //block delta = prng.get();
    //auto type = OTType::Correlated;

    std::vector<std::array<block, 2>> msg2(n);
    std::vector<block> msg1(n);
    BitVector choice(n);

    auto p0 = sender.silentSend(msg2, prng, sockets[0]);
    auto p1 = recver.silentReceive(choice, msg1, prng, sockets[1]);


    eval(p0, p1);

    checkRandom(msg1, msg2, choice, n, verbose);
#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}



void OtExt_Silent_mal_Test(const oc::CLP& cmd)
{

#ifdef ENABLE_SILENTOT


    
    auto sockets = cp::LocalAsyncSocket::makePair();

    u64 n = 12093;//

    bool verbose = cmd.getOr("v", 0) > 1;

    PRNG prng(toBlock(cmd.getOr("seed", 0)));
    PRNG prng1(toBlock(cmd.getOr("seed1", 1)));

    SilentOtExtSender sender;
    SilentOtExtReceiver recver;

    sender.mMalType = SilentSecType::Malicious;
    recver.mMalType = SilentSecType::Malicious;

    std::vector<std::array<block, 2>> msg2(n);
    std::vector<block> msg1(n);
    BitVector choice(n);

    auto p0 = sender.silentSend(msg2, prng, sockets[0]);
    auto p1 = recver.silentReceive(choice, msg1, prng, sockets[1]);


    eval(p0, p1);

    checkRandom(msg1, msg2, choice, n, verbose);
#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}

void Tools_Pprf_test(const CLP& cmd)
{
#if defined(ENABLE_SILENTOT) || defined(ENABLE_SILENT_VOLE)

    u64 depth = cmd.getOr("d", 3);;
    u64 domain = 1ull << depth;
    auto threads = cmd.getOr("t", 3ull);
    u64 numPoints = cmd.getOr("s", 8);

    PRNG prng(ZeroBlock);

    //IOService ios;
    //Session s0(ios, "localhost:1212", SessionMode::Server);
    //Session s1(ios, "localhost:1212", SessionMode::Client);
    //auto sockets[0] = s0.addChannel();
    //auto sockets[1] = s1.addChannel();
    
    auto sockets = cp::LocalAsyncSocket::makePair();


    auto format = PprfOutputFormat::Plain;
    SilentMultiPprfSender sender;
    SilentMultiPprfReceiver recver;

    sender.configure(domain, numPoints);
    recver.configure(domain, numPoints);

    auto numOTs = sender.baseOtCount();
    std::vector<std::array<block, 2>> sendOTs(numOTs);
    std::vector<block> recvOTs(numOTs);
    BitVector recvBits = recver.sampleChoiceBits(domain, format, prng);

    prng.get(sendOTs.data(), sendOTs.size());
    //sendOTs[cmd.getOr("i",0)] = prng.get();

    //recvBits[16] = 1;
    for (u64 i = 0; i < numOTs; ++i)
    {
        //recvBits[i] = 0;
        recvOTs[i] = sendOTs[i][recvBits[i]];
    }
    sender.setBase(sendOTs);
    recver.setBase(recvOTs);

    Matrix<block> sOut(domain, numPoints);
    Matrix<block> rOut(domain, numPoints);
    std::vector<u64> points(numPoints);
    recver.getPoints(points, format);

    auto p0 = sender.expand(sockets[0], {&CCBlock,1}, prng, sOut, format, true, threads);
    auto p1 = recver.expand(sockets[1], prng, rOut, format, true, threads);

    eval(p0, p1);

    bool failed = false;


    for (u64 j = 0; j < numPoints; ++j)
    {

        for (u64 i = 0; i < domain; ++i)
        {

            auto exp = sOut(i, j);
            if (points[j] == i)
                exp = exp ^ CCBlock;

            if (neq(exp, rOut(i, j)))
            {
                failed = true;

                if (cmd.isSet("v"))
                    std::cout << Color::Red;
            }
            if (cmd.isSet("v"))
                std::cout << "r[" << j << "][" << i << "] " << exp << " " << rOut(i, j) << std::endl << Color::Default;
        }
    }

    if (failed)
        throw RTE_LOC;

#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}

void Tools_Pprf_trans_test(const CLP& cmd)
{
#if defined(ENABLE_SILENTOT) || defined(ENABLE_SILENT_VOLE)

    //u64 depth = 6;
    //u64 domain = 13;// (1ull << depth) - 7;
    //u64 numPoints = 40;

    u64 domain = cmd.getOr("d", 334);
    auto threads = cmd.getOr("t", 3ull);
    u64 numPoints = cmd.getOr("s", 5) * 8;
    //bool mal = cmd.isSet("mal");

    PRNG prng(ZeroBlock);

    
    auto sockets = cp::LocalAsyncSocket::makePair();




    auto format = PprfOutputFormat::InterleavedTransposed;
    SilentMultiPprfSender sender;
    SilentMultiPprfReceiver recver;

    sender.configure(domain, numPoints);
    recver.configure(domain, numPoints);

    auto numOTs = sender.baseOtCount();
    std::vector<std::array<block, 2>> sendOTs(numOTs);
    std::vector<block> recvOTs(numOTs);
    BitVector recvBits = recver.sampleChoiceBits(domain * numPoints, format, prng);
    //recvBits.randomize(prng);

    //recvBits[16] = 1;
    prng.get(sendOTs.data(), sendOTs.size());
    for (u64 i = 0; i < numOTs; ++i)
    {
        //recvBits[i] = 0;
        recvOTs[i] = sendOTs[i][recvBits[i]];
    }
    sender.setBase(sendOTs);
    recver.setBase(recvOTs);

    auto cols = (numPoints * domain + 127) / 128;
    Matrix<block> sOut(128, cols);
    Matrix<block> rOut(128, cols);

    std::vector<u64> points(numPoints);
    recver.getPoints(points, format);




    auto p0 = sender.expand(sockets[0], { &AllOneBlock,1 }, prng, sOut, format, true, threads);
    auto p1 = recver.expand(sockets[1], prng, rOut, format, true, threads);


    eval(p0, p1);
    bool failed = false;

    Matrix<block> out(128, cols);
    Matrix<block> outT(numPoints * domain, 1);

    if (cmd.getOr("v", 0) > 1)
        std::cout << sender.mDomain << " " << sender.mPntCount <<
        " " << sOut.rows() << " " << sOut.cols() << std::endl;

    for (u64 i = 0; i < cols; ++i)
    {
        for (u64 j = 0; j < 128; ++j)
        {
            out(j, i) = (sOut(j, i) ^ rOut(j, i));
            //if (cmd.isSet("v"))
            //	std::cout << "r[" << i << "][" << j << "] " << out(j,i)  << " ~ " << rOut(j, i) << std::endl << Color::Default;
        }
    }
    transpose(MatrixView<block>(out), MatrixView<block>(outT));

    for (u64 i = 0; i < outT.rows(); ++i)
    {

        auto f = std::find(points.begin(), points.end(), i) != points.end();

        auto exp = f ? AllOneBlock : ZeroBlock;

        if (neq(outT(i), exp))
        {
            failed = true;

            if (cmd.getOr("v", 0) > 1)
                std::cout << Color::Red;
        }
        if (cmd.getOr("v", 0) > 1)
            std::cout << i << " " << outT(i) << " " << exp << std::endl << Color::Default;
    }

    if (failed)
        throw RTE_LOC;

#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}


void Tools_Pprf_inter_test(const CLP& cmd)
{
#if defined(ENABLE_SILENTOT) || defined(ENABLE_SILENT_VOLE)

    //u64 depth = 6;
    //u64 domain = 13;// (1ull << depth) - 7;
    //u64 numPoints = 40;

    u64 domain = cmd.getOr("d", 334);
    auto threads = cmd.getOr("t", 3ull);
    u64 numPoints = cmd.getOr("s", 5) * 8;
    //bool mal = cmd.isSet("mal");

    PRNG prng(ZeroBlock);

    auto sockets = cp::LocalAsyncSocket::makePair();


    auto format = PprfOutputFormat::Interleaved;
    SilentMultiPprfSender sender;
    SilentMultiPprfReceiver recver;

    sender.configure(domain, numPoints);
    recver.configure(domain, numPoints);

    auto numOTs = sender.baseOtCount();
    std::vector<std::array<block, 2>> sendOTs(numOTs);
    std::vector<block> recvOTs(numOTs);
    BitVector recvBits = recver.sampleChoiceBits(domain * numPoints, format, prng);
    //recvBits.randomize(prng);

    //recvBits[16] = 1;
    prng.get(sendOTs.data(), sendOTs.size());
    for (u64 i = 0; i < numOTs; ++i)
    {
        //recvBits[i] = 0;
        recvOTs[i] = sendOTs[i][recvBits[i]];
    }
    sender.setBase(sendOTs);
    recver.setBase(recvOTs);

    //auto cols = (numPoints * domain + 127) / 128;
    Matrix<block> sOut2(numPoints * domain, 1);
    Matrix<block> rOut2(numPoints * domain, 1);
    std::vector<u64> points(numPoints);
    recver.getPoints(points, format);


    auto p0 = sender.expand(sockets[0], { &AllOneBlock,1 }, prng, sOut2, format, true, threads);
    auto p1 = recver.expand(sockets[1], prng, rOut2, format, true, threads);

    eval(p0, p1);
    for (u64 i = 0; i < rOut2.rows(); ++i)
    {
        sOut2(i) = (sOut2(i) ^ rOut2(i));
    }


    bool failed = false;
    for (u64 i = 0; i < sOut2.rows(); ++i)
    {

        auto f = std::find(points.begin(), points.end(), i) != points.end();

        auto exp = f ? AllOneBlock : ZeroBlock;

        if (neq(sOut2(i), exp))
        {
            failed = true;

            if (cmd.getOr("v", 0) > 1)
                std::cout << Color::Red;
        }
        if (cmd.getOr("v", 0) > 1)
            std::cout << i << " " << sOut2(i) << " " << exp << std::endl << Color::Default;
    }

    if (failed)
        throw RTE_LOC;


#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}



void Tools_Pprf_blockTrans_test(const oc::CLP& cmd)
{
#if defined(ENABLE_SILENTOT) || defined(ENABLE_SILENT_VOLE)


    u64 depth = cmd.getOr("d", 2);;
    u64 domain = 1ull << depth;
    auto threads = cmd.getOr("t", 1ull);
    u64 numPoints = cmd.getOr("s", 8);

    PRNG prng(ZeroBlock);

    auto sockets = cp::LocalAsyncSocket::makePair();


    auto format = PprfOutputFormat::BlockTransposed;
    SilentMultiPprfSender sender;
    SilentMultiPprfReceiver recver;

    sender.configure(domain, numPoints);
    recver.configure(domain, numPoints);

    auto numOTs = sender.baseOtCount();
    std::vector<std::array<block, 2>> sendOTs(numOTs);
    std::vector<block> recvOTs(numOTs);
    BitVector recvBits = recver.sampleChoiceBits(domain, format, prng);

    prng.get(sendOTs.data(), sendOTs.size());
    //sendOTs[cmd.getOr("i",0)] = prng.get();

    //recvBits[16] = 1;
    for (u64 i = 0; i < numOTs; ++i)
    {
        //recvBits[i] = 0;
        recvOTs[i] = sendOTs[i][recvBits[i]];
    }
    sender.setBase(sendOTs);
    recver.setBase(recvOTs);

    Matrix<block> sOut(numPoints, domain);
    Matrix<block> rOut(numPoints, domain);
    std::vector<u64> points(numPoints);
    recver.getPoints(points, format);

    cp::sync_wait(cp::when_all_ready(
        sender.expand(sockets[0], span<block>{}, prng, sOut, format, false, threads),
        recver.expand(sockets[1], prng, rOut, format, false, threads)
    ));

    bool failed = false;

    for (u64 j = 0; j < numPoints; ++j)
    {

        for (u64 i = 0; i < domain; ++i)
        {
            auto ss = sOut(j, i);
            auto rr = rOut(j, i);

            if (points[j] == i)
            {
                if (ss == rr || rr != ZeroBlock)
                {
                    failed = true;

                    if (cmd.isSet("v"))
                        std::cout << Color::Red;
                }

            }
            else
            {
                if (ss != rr  || rr == ZeroBlock)
                {
                    failed = true;

                    if (cmd.isSet("v"))
                        std::cout << Color::Red;
                }
            }
            if (cmd.isSet("v"))
                std::cout << "r[" << j << "][" << i << "] " << ss << " " << rr << std::endl << Color::Default;
        }
    }

    if (failed)
        throw RTE_LOC;

#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}



void Tools_Pprf_callback_test(const oc::CLP& cmd)
{
#if defined(ENABLE_SILENTOT) || defined(ENABLE_SILENT_VOLE)

    u64 domain = cmd.getOr("d", 512);
    auto threads = cmd.getOr("t", 1ull);
    u64 numPoints = cmd.getOr("s", 5) * 8;

    PRNG prng(ZeroBlock);
    auto sockets = cp::LocalAsyncSocket::makePair();


    auto format = PprfOutputFormat::Callback;
    SilentMultiPprfSender sender;
    SilentMultiPprfReceiver recver;

    sender.configure(domain, numPoints);
    recver.configure(domain, numPoints);

    auto numOTs = sender.baseOtCount();
    std::vector<std::array<block, 2>> sendOTs(numOTs);
    std::vector<block> recvOTs(numOTs);
    BitVector recvBits = recver.sampleChoiceBits(domain * numPoints, format, prng);

    prng.get(sendOTs.data(), sendOTs.size());
    for (u64 i = 0; i < numOTs; ++i)
    {
        recvOTs[i] = sendOTs[i][recvBits[i]];
    }
    sender.setBase(sendOTs);
    recver.setBase(recvOTs);

    //auto cols = (numPoints * domain + 127) / 128;
    Matrix<block> sOut2(numPoints * domain, 1);
    Matrix<block> rOut2(numPoints * domain, 1);
    std::vector<u64> points(numPoints);
    recver.getPoints(points, format);

    sender.mOutputFn = [&](u64 treeIdx, span<AlignedArray<block, 8>> data)
    {
        span<block> d = sOut2;
        d = d.subspan(treeIdx * data.size());
        d = d.subspan(0, std::min<u64>(d.size(), data.size() * 8));
        memcpy(d.data(), data.data(), d.size_bytes());
    };
    recver.mOutputFn = [&](u64 treeIdx, span<AlignedArray<block, 8>> data)
    {
        span<block> d = rOut2;
        d = d.subspan(treeIdx * data.size());
        d = d.subspan(0, std::min<u64>(d.size(), data.size() * 8));
        memcpy(d.data(), data.data(), d.size_bytes());
    };


    auto p0 = sender.expand(sockets[0], { &AllOneBlock,1 }, prng, span<block>{}, format, true, threads);
    auto p1 = recver.expand(sockets[1], prng, span<block>{}, format, true, threads);

    eval(p0, p1);
    for (u64 i = 0; i < rOut2.rows(); ++i)
    {
        sOut2(i) = (sOut2(i) ^ rOut2(i));
    }

    bool failed = false;
    for (u64 i = 0; i < sOut2.rows(); ++i)
    {

        auto f = std::find(points.begin(), points.end(), i) != points.end();

        auto exp = f ? AllOneBlock : ZeroBlock;

        if (neq(sOut2(i), exp))
        {
            failed = true;

            if (cmd.getOr("v", 0) > 1)
                std::cout << Color::Red;
        }
        if (cmd.getOr("v", 0) > 1)
            std::cout << i << " " << sOut2(i) << " " << exp << std::endl << Color::Default;
    }

    if (failed)
        throw RTE_LOC;

#else
    throw UnitTestSkipped("ENABLE_SILENTOT not defined.");
#endif
}
