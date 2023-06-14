use proc_macro::TokenStream;
use quote::{format_ident, quote};

use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Expr, ExprReference, Token, Type};

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
    let sub_types = args.sub_types;

    let field_idents: Vec<_> = (0..sub_types.len())
        .map(|idx| format_ident!("remote_receiver_{}", idx))
        .collect();

    let sub_sender_idents: Vec<_> = (0..sub_types.len())
        .map(|idx| format_ident!("sub_sender_{}", idx))
        .collect();

    let output = quote! {
        {
            async {
                #(
                let (#sub_sender_idents, #field_idents) = ::mpc_channel::channel::<#sub_types, 128>(#local_buffer);
                )*

                let receivers = (
                    #(#field_idents),*
                );
                #sender.send(receivers).await?;
                let msg = #receiver.recv().await?.ok_or(::mpc_channel::CommunicationError::RemoteClosed)?;
                let (
                    #(#field_idents),*
                ) = msg;
                Ok::<_, ::mpc_channel::CommunicationError>((#((#sub_sender_idents, #field_idents)),*))
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
