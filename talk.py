import whisper
import gradio as gr 
import time
from pyChatGPT import ChatGPT
import warnings
import numpy as np
import os


warnings.filterwarnings("ignore")

secret_token = "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..Imxz7CoiM6j21-4M.I9YGkaZiggIi6eUJ6AF_ZxykKucD_RB-Kt5e5O08_GJNRzeqa044VcN1gG6nFdxZYX6p5RLC0GtQydDjfGexV_oQ6NbNJAJsbRltY4l126czwagmjyx6qLrfbM26RD1f6SYuSI81VwILp82xxu6ObtVktmYXukDfJnVThQjrmxXneguKUZdCLlMk6umQSysJQFJWtUMaV2zhK2Nec_yDgmhn98jwQ9Jzal_uW40YEnZMV-3CRk1o79HccOCo6aCaK4EXjOCB-r9A-w_5eoxkbJS9JsakO5cwIQ8DhZdRdarjR6Sb14TWZXYcPD8cMOq8cJV7ZaEqmRzsrzmQFR9dzifyrXjsf4B6R5-mZzY-BDn2jr2nAgYJh--h0Bxp-4JDz6ewYEKLWiTQcx3E9GTGcHji-URjvDlOCZgydGin6GZWxdglglVsZrHhaPpl0YOt9t2lTXQFdD2Jg0bR5e8uQ6_KjwvKt-Y7zxSgF0UlWLTufTkS737nIZ6MTIU9ZjJV6aYLW5nkFrsJGiovQ0bKsHK2aX8AHLH3eYEuGUIkkm6dOyqvj2kcwOx2U9NJVGyZi8AybWbAStcQ8ND117t1BQVMFGTVLLSnSYWCNsDD2pJul1aZkrun3JAvb59pGNtNuZ-RN8ZP60F-CiZBshbBS-ok9S8vgJgavRMYYasqEuajcby1Wku-2gPuKaMw1UZhNmcFFmUG6kLh_URvL6AwiMQKxZu5ByDHRDEVtFbZSEYcoWa8Z3-2JoM_KcguhCCW9tox9UHwll4E_3qr8HSrQKsJ3mjlhCppw833IOcCl3_lwB-AWPq6YPT4WqAWVqWk9w59QZrPw-FgQUgM9nDqiZ7mVWHYeuSAqh0D-aAE6zLuJyaKskI9yd8IV1DLTnGrvfAicj5tddnz1hF78xFWk2J09jj7HgmuyY-lm4ex7JK7ujq2JN1ibFZzA1nI2Is1z95opBC5ujYT5DqDFn5OcJYjZohr1Nuej1kcJEC3mLdluIY_lOYo2MDL7MMfcKOWdyLcN1EcK3kUSLEXZwux_2of-MOS5_5gHv6OM-Kzpk9VuG9_LVCD1Cuk2IfydnVJAorqXOwtGQ2M8Qbfma7_buCbzfLlVqjZU2pcP8SyoKaxneaSKYQZ-zkSxwzaNa34jZTveM-duZrRtseKtRgVVnXr3Exje3P66gv53XtaJS4V-mLXCDujzv2ssQe1GH4aabxUlwjWKiFL89Z9XhpM32zQGm0rV6DmKowCkxf45_nW6xO_nqo0EGGFLK2jLmuOlNRhXRihnyh2q_1HXDI2CpyMnkv87BJ6ofpcAo8vv7CWdK3liLSqTzWsEmSHWaF2D-3mhPICnR31hVhCQZoXPzwrI-7UNIP6hjDdm8xyYKvq7i8y-LgMolzFjE71RsYA8j_2w49gFbRHSSZwGhJTS7eXExIkoxTBsI9ruRsq3epLQ1ec3IVhcX3udbwDX_olhbK_JLTy2Bs4qgl-w1TRi6mX5Kk1oZfOTswcmtiKKXOEIQAl9iCRhpeIYf_Kq-dvtm14J2rR-px8Dp-wio2nYIJ2zIKgQHFVljM_XfGD8xB6lXrjSrScR4JBVdM09oSODJE8PGFvQ0SX9W7oyvwXafjstb23mkS0RCnChM4prHKwjKI4hqscYijGPnOxRgAJK9C4db83WcRHdW4nH4GNReu4AevhGQwpJSEEY72Vx-fX38FQVlvQNt248y1rbYkySKlFDGhAH1MFYy0hb8C4Dj2_--h3PlsuJLP7c8EtK2VUxyeq9sOADDMipubWOuBvLBtMOJNCllUin5y2uryDZm1ckYP38s3IDrCQabi3CtJtikCNsBuc5BkFTU_zY19VFj0MnK4MevBsLZZgwscMAirgvnmweuwwkBMA3ENXEDBAwWZMpTcQiMyXr0XmogZkIklB05IhmKJ8eZDi44bB50WaACCXHH-RN-bf_tKzBkh-tHt6C0sEirmwuZ4t8XtE3DoXL6TCEQrEpUgMJ0bO48n-1fAsf6DikFtup_fL9k2DRHaEqkVxv7UFXEdkGyJSauYer4yCrtMLSWwGcrg2Mmw-tnm4IAA6-58z2vpZowumm1QSC97m--fgE5idB7oLgx688hAkJ3qvk1ixKzLHiXPNRNVhu4C_w0AR19tGPS5RpjnQBzGSA4ssGG-akMEEWUyV7hc5gMe_olY4mUH3W_Counfl5Y7Qron1S6_tlGfxCZO4S4PTNgfK7efoGveQG6gPv8j8i-0Kiky8qXCd3vMtLGpsVfYcF5fRc6fF8nZkNpVKb6vSLXksiF8TrE3MS1jiUuJ-P6b6PF5BewNQqTZT0g6IRA2eiVwU3RQdTWxUxUdZwM73bJsPJTqscMrj3wHB-2fjajVhpaHXBdg59mESwQp5wtlio27sYw5X6VIrAL8-awJVqjm9mNKLOz48uwbOn5J2C_Q6zecA4rbJ2iqzZNjCabeijtFED9lCvpdt99FAk4_L5ilvQYCKqJ8NjAb19373J0b0lEq9lJlpcxz8Kdo5zvTj-Fd5VM4VyITJxMaa-aYdx3Gg.0bSAlAcXTOHyk2x742bLzw"

model = whisper.load_model("base")

model.device

def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
 
    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    result_text = result.text

    print(result_text)


    # Pass the generated text to Audio
    chatgpt_api = ChatGPT(secret_token)
    resp = chatgpt_api.send_message(result_text)
    out_result = resp['message']

    return [result_text, out_result]



output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="ChatGPT Output")

auth_token = os.getenv("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")

gr.Interface(
    title = 'talkWIth By Irfan',
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],

    outputs=[
        output_1,  output_2
    ],
    live=True).launch()
