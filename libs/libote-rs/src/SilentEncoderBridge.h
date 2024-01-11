#pragma once
#include <libOTe/Tools/LDPC/LdpcEncoder.h>
#include <libOTe/Tools/EACode/EACode.h>
#include <libOTe/Tools/ExConvCode/ExConvCode.h>
#include <cryptoTools/Common/Defines.h>
#include <rust/cxx.h>

namespace osuCryptoBridge {
    using SilverCodeWeight = osuCrypto::SilverCode::code;

    struct SilverEncBridge : public oc::SilverEncoder {

        void dualEncodeBlock(rust::Slice<oc::block> c) {
            dualEncode(oc::span<oc::block>(c.begin(), c.end()));
        }

        void dualEncode2Block(rust::Slice<oc::block> c0, rust::Slice<oc::u8> c1) {
            oc::span<oc::block> c0_span(c0.begin(), c0.end());
            oc::span<oc::u8> c1_span(c1.begin(), c1.end());
            dualEncode2(c0_span, c1_span);
        }
    };

    std::unique_ptr<SilverEncBridge> newEnc() {
        return std::make_unique<SilverEncBridge>();
    }

    struct EACodeBridge : public oc::EACode {
        void config(
                oc::u64 messageSize,
                oc::u64 codeSize,
                oc::u64 expanderWeight) {
            EACode::config(messageSize, codeSize, expanderWeight);
        }

        void dualEncodeBlock(rust::Slice<oc::block> e,rust::Slice<oc::block> w) {
            oc::span<oc::block> e_span(e.begin(), e.end());
            oc::span<oc::block> w_span(w.begin(), w.end());
            dualEncode(e_span, w_span);
        }

        void dualEncode2Block(rust::Slice<oc::block> e0, rust::Slice<oc::block> w0, rust::Slice<oc::u8> e1, rust::Slice<oc::u8> w1) {
            oc::span<oc::block> e0_span(e0.begin(), e0.end());
            oc::span<oc::block> w0_span(w0.begin(), w0.end());
            oc::span<oc::u8> e1_span(e1.begin(), e1.end());
            oc::span<oc::u8> w1_span(w1.begin(), w1.end());
            dualEncode2(e0_span, w0_span, e1_span, w1_span);
        }
    };

    std::unique_ptr<EACodeBridge> newEACode() {
        return std::make_unique<EACodeBridge>();
    }

    struct ExConvCodeBridge: public oc::ExConvCode {
        void config(
                oc::u64 messageSize,
                oc::u64 codeSize,
                oc::u64 expanderWeight,
                oc::u64 accumulatorSize) {
            ExConvCode::config(messageSize, codeSize, expanderWeight, accumulatorSize);
        }

        void dualEncodeBlock(rust::Slice<oc::block> e) {
            dualEncode(oc::span<oc::block>(e.begin(), e.end()));
        }

        void dualEncode2Block(rust::Slice<oc::block> e0, rust::Slice<oc::u8> e1) {
            oc::span<oc::block> e0_span(e0.begin(), e0.end());
            oc::span<oc::u8> e1_span(e1.begin(), e1.end());
            dualEncode2(e0_span, e1_span);
        }
    };

    std::unique_ptr<ExConvCodeBridge> newExConvCode() {
        return std::make_unique<ExConvCodeBridge>();
    }
}

