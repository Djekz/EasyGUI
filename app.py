from rvcvtwo import *
import yt_dlp
import subprocess, torch, os, traceback, sys, warnings, shutil, numpy as np
from mega import Mega
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import threading
from time import sleep
from subprocess import Popen
import faiss
from random import shuffle
import json, datetime, requests
from gtts import gTTS
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from i18n import I18nAuto

import signal

import math

from utils import load_audio, CSVutil

global DoFormant, Quefrency, Timbre

def downloaderyt(url, audio_name, audio_format):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
        }],
        'outtmpl': f"audios/{audio_name}",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_audio(video_url, audio_name, audio_format):
    downloaderyt(video_url, audio_name, audio_format)
    return f"Audio downloaded with name {audio_name} and format {audio_format}"

with gr.Blocks(theme='HRVC/Interstellar', title='Mangio-RVC-Web 💻') as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML("<h1>  EasyGUI   </h1>")    
            gr.HTML("<h10>   Easy GUI coded by Rejekt's   </h10>")
            
            # Inference Preset Row
            # with gr.Row():
            #     mangio_preset = gr.Dropdown(label="Inference Preset", choices=sorted(get_presets()))
            #     mangio_preset_name_save = gr.Textbox(
            #         label="Your preset name"
            #     )
            #     mangio_preset_save_btn = gr.Button('Save Preset', variant="primary")

            # Other RVC stuff
            with gr.Row():
                sid0 = gr.Dropdown(label="1.Choose your Model.", choices=sorted(names), value=check_for_name())
                refresh_button = gr.Button("Refresh", variant="primary")
                if check_for_name() != '':
                    get_vc(sorted(names)[0])
                vc_transform0 = gr.Number(label="Optional: You can change the pitch here or leave it at 0.", value=0)
                #clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                #clean_button.click(fn=clean, inputs=[], outputs=[sid0])
                sid0.change(
                    fn=get_vc,
                    inputs=[sid0],
                    outputs=[spk_item],
                )
                but0 = gr.Button("Convert", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.File(label="Drop your audio here & hit the Reload button.")
                    with gr.Row():
                        record_button=gr.Audio(source="microphone", label="OR Record audio.", type="filepath")
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label="2.Choose your audio.",
                            value="./audios/someguy.mp3",
                            choices=audio_files
                            )
                        dropbox.upload(fn=save_to_wav2, inputs=[dropbox], outputs=[input_audio0])
                        dropbox.upload(fn=change_choices2, inputs=[], outputs=[input_audio0])
                        refresh_button2 = gr.Button("Refresh", variant="primary", size='sm')
                        record_button.change(fn=save_to_wav, inputs=[record_button], outputs=[input_audio0])
                        record_button.change(fn=change_choices2, inputs=[], outputs=[input_audio0])
                    with gr.Row():
                        with gr.Accordion('Text To Speech', open=False):
                            with gr.Column():
                                lang = gr.Radio(label='Chinese & Japanese do not work with ElevenLabs currently.',choices=['en','es','fr','pt','zh-CN','de','hi','ja'], value='en')
                                api_box = gr.Textbox(label="Enter your API Key for ElevenLabs, or leave empty to use GoogleTTS", value='')
                                elevenid=gr.Dropdown(label="Voice:", choices=eleven_voices)
                            with gr.Column():
                                tfs = gr.Textbox(label="Input your Text", interactive=True, value="This is a test.")
                                tts_button = gr.Button(value="Speak")
                                tts_button.click(fn=elevenTTS, inputs=[api_box,tfs, elevenid, lang], outputs=[record_button, input_audio0])
                    
                with gr.Column():
                    with gr.Accordion("Index Settings", open=False):
                        file_index1 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=True,
                            )
                        sid0.change(fn=match_index, inputs=[sid0],outputs=[file_index1])
                        refresh_button.click(
                            fn=change_choices, inputs=[], outputs=[sid0, file_index1]
                            )
                        # file_big_npy1 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=0.66,
                            interactive=True,
                            )
                    vc_output2 = gr.Audio(
                        label="Output Audio (Click on the Three Dots in the Right Corner to Download)",
                        type='filepath',
                        interactive=False,
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        f0method0 = gr.Radio(
                            label="Optional: Change the Pitch Extraction Algorithm.\nExtraction methods are sorted from 'worst quality' to 'best quality'.\nmangio-crepe may or may not be better than rmvpe in cases where 'smoothness' is more important, but rmvpe is the best overall.",
                            choices=["dio", "crepe-tiny", "mangio-crepe-tiny", "crepe", "harvest", "mangio-crepe", "rmvpe"], # Fork Feature. Add Crepe-Tiny
                            value="rmvpe",
                            interactive=True,
                        )
                        
                        crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="Mangio-Crepe Hop Length. Higher numbers will reduce the chance of extreme pitch changes but lower numbers will increase accuracy. 64-192 is a good range to experiment with.",
                            value=120,
                            interactive=True,
                            visible=False,
                            )
                        f0method0.change(fn=whethercrepeornah, inputs=[f0method0], outputs=[crepe_hop_length])
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                            )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                            )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0.21,
                            interactive=True,
                            )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                            )
                        formanting = gr.Checkbox(
                            value=bool(DoFormant),
                            label="[EXPERIMENTAL] Formant shift inference audio",
                            info="Used for male to female and vice-versa conversions",
                            interactive=True,
                            visible=True,
                        )
                        
                        formant_preset = gr.Dropdown(
                            value='',
                            choices=get_fshift_presets(),
                            label="browse presets for formanting",
                            visible=bool(DoFormant),
                        )
                        formant_refresh_button = gr.Button(
                            value='\U0001f504',
                            visible=bool(DoFormant),
                            variant='primary',
                        )
                        #formant_refresh_button = ToolButton( elem_id='1')
                        #create_refresh_button(formant_preset, lambda: {"choices": formant_preset}, "refresh_list_shiftpresets")
                        
                        qfrency = gr.Slider(
                                value=Quefrency,
                                info="Default value is 1.0",
                                label="Quefrency for formant shifting",
                                minimum=0.0,
                                maximum=16.0,
                                step=0.1,
                                visible=bool(DoFormant),
                                interactive=True,
                            )
                        tmbre = gr.Slider(
                            value=Timbre,
                            info="Default value is 1.0",
                            label="Timbre for formant shifting",
                            minimum=0.0,
                            maximum=16.0,
                            step=0.1,
                            visible=bool(DoFormant),
                            interactive=True,
                        )
                        
                        formant_preset.change(fn=preset_apply, inputs=[formant_preset, qfrency, tmbre], outputs=[qfrency, tmbre])
                        frmntbut = gr.Button("Apply", variant="primary", visible=bool(DoFormant))
                        formanting.change(fn=formant_enabled,inputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button],outputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button])
                        frmntbut.click(fn=formant_apply,inputs=[qfrency, tmbre], outputs=[qfrency, tmbre])
                        formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset, qfrency, tmbre],outputs=[formant_preset, qfrency, tmbre])
            with gr.Row():
                vc_output1 = gr.Textbox("")
                f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
                
                but0.click(
                    vc_single,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        # file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        crepe_hop_length
                    ],
                    [vc_output1, vc_output2],
                )
                        
            
        with gr.TabItem("Download Model"):
            with gr.Row():
                url=gr.Textbox(label="Enter the URL to the Model:")
            with gr.Row():
                model = gr.Textbox(label="Name your model:")
                download_button=gr.Button("Download")
            with gr.Row():
                status_bar=gr.Textbox(label="")
                download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])
            with gr.Row():
                gr.Markdown(
                """
                Original RVC:https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI Mangio’s RVC Fork:https://github.com/Mangio621/Mangio-RVC-Fork ❤️ If you like the EasyGUI, help me keep it.❤️ https://paypal.me/lesantillan
                """
                )
        with gr.TabItem("Download acapella"):
            with gr.Row():
                url=gr.Textbox(label="Enter the URL to the audio youtube:")
            with gr.Row():
                name_audio = gr.Textbox(label="Name your audio (no space):")
                format_audio4 = gr.Radio(label="Select the output format", choices=["wav", "mp3"])
                status_bar=gr.Textbox(label="")
                download_button=gr.Button("Download")
            with gr.Row():
                download_button.click(fn=downloaderyt, inputs=[url, name_audio, format_audio4], outputs=[status_bar])
            with gr.Row():
                gr.Markdown(
                """
                yt_dlp base code by me (blane187)
                """
                )

        

        
            
                
    app.queue(concurrency_count=511, max_size=1022).launch(share=True, debug=True)
#endregion
