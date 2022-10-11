use proc_macro::TokenStream;
use quote::{format_ident, quote};

use syn::parse::{Parse, ParseStream};
use syn::token::Token;
use syn::{parse_macro_input, parse_quote, Expr, ExprReference, Token, Type, Variant};

struct Args {
    sender: ExprReference,
    receiver: ExprReference,
    local_buffer: Expr,
    sub_types: Vec<Type>,
}

#[proc_macro]
pub fn sub_channels_for(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as Args);
    let sender = args.sender.expr;
    let receiver = args.receiver.expr;
    let local_buffer = args.local_buffer;

    let variant_idents: Vec<_> = (0..args.sub_types.len())
        .map(|idx| format_ident!("receivers_{}", idx))
        .collect();
    let receivers_variants = args
        .sub_types
        .iter()
        .zip(&variant_idents)
        .map::<Variant, _>(|(ty, variant_ident)| {
            parse_quote!(
                #variant_ident(::mpc_channel::Receiver<#ty>)
            )
        });
    let receivers_enum = quote! {
        #[derive(::serde::Serialize, ::serde::Deserialize)]
        enum __Receivers {
            #(#receivers_variants),*
        }
    };

    let sub_sender_idents: Vec<_> = (0..args.sub_types.len())
        .map(|idx| format_ident!("sub_sender_{}", idx))
        .collect();
    let remote_sub_receiver_idents: Vec<_> = (0..args.sub_types.len())
        .map(|idx| format_ident!("remote_sub_receiver_{}", idx))
        .collect();
    let sub_receiver_idents: Vec<_> = (0..args.sub_types.len())
        .map(|idx| format_ident!("sub_receiver_{}", idx))
        .collect();

    let output = quote! {
        {
            #receivers_enum

            async {
                #(
                let (#sub_sender_idents, #remote_sub_receiver_idents) = ::mpc_channel::channel(#local_buffer);
                )*
                #(
                #sender.send(__Receivers::#variant_idents(#remote_sub_receiver_idents)).await?;
                )*
                #(
                let msg = #receiver.recv().await?.ok_or(::mpc_channel::CommunicationError::RemoteClosed)?;
                let #sub_receiver_idents = match msg {
                    __Receivers::#variant_idents(recv) => recv,
                    _ => Err(::mpc_channel::CommunicationError::UnexpectedMessage)?,
                };
                )*
                Ok::<_, ::mpc_channel::CommunicationError>((#((#sub_sender_idents, #sub_receiver_idents)),*))
            }
        }

    };

    output.into()
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let sender = input.parse()?;
        input.parse::<Token![,]>()?;
        let receiver = input.parse()?;
        input.parse::<Token![,]>()?;
        let local_buffer = input.parse()?;
        input.parse::<Token![,]>()?;
        let sub_types = input.parse_terminated::<_, Token![,]>(Type::parse)?;
        let sub_types = sub_types.into_iter().collect();
        Ok(Self {
            sender,
            receiver,
            local_buffer,
            sub_types,
        })
    }
}
