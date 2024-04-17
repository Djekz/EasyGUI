from rvctwo import *

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



with gr.Blocks(theme=gr.themes.Base(), title='Mangio-RVC-Web 💻') as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML("<h1>  EasyGUI Huggingface Version   </h1>")    
            gr.HTML("<h4>   Inference may take time because this space does not use GPU :(  </h4>")
            gr.HTML("<h10>   Easy GUI coded by Rejekt's   </h10>")
            gr.HTML("<h4>  If you want to use this space privately, I recommend you duplicate the space.  </h4>")

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
                    with gr.Row():
                        with gr.Accordion('Wav2Lip', open=False):
                            with gr.Row():
                                size = gr.Radio(label='Resolution:',choices=['Half','Full'])
                                face = gr.UploadButton("Upload A Character",type='file')
                                faces = gr.Dropdown(label="OR Choose one:", choices=['None','Ben Shapiro','Andrew Tate'])
                            with gr.Row():
                                preview = gr.Textbox(label="Status:",interactive=False)
                                face.upload(fn=success_message,inputs=[face], outputs=[preview, faces])
                            with gr.Row():
                                animation = gr.Video(type='filepath')
                                refresh_button2.click(fn=change_choices2, inputs=[], outputs=[input_audio0, animation])
                            with gr.Row():
                                animate_button = gr.Button('Animate')

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
                    animate_button.click(fn=mouth, inputs=[size, face, vc_output2, faces], outputs=[animation, preview])
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
                        
            with gr.Accordion("Batch Conversion",open=False):
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                        )
                        opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                        f0method1 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                    with gr.Column():
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    with gr.Column():
                        dir_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                            value="E:\codes\py39\\test-20230416b\\todo-songs",
                        )
                        inputs = gr.File(
                            file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                        )
                    with gr.Row():
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("转换"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("输出信息"))
                    but1.click(
                        vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                            crepe_hop_length,
                        ],
                        [vc_output3],
                    )
                    but1.click(fn=lambda: easy_uploader.clear())
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

        def has_two_files_in_pretrained_folder():
            pretrained_folder = "./pretrained/"
            if not os.path.exists(pretrained_folder):
                return False

            files_in_folder = os.listdir(pretrained_folder)
            num_files = len(files_in_folder)
            return num_files >= 2

        if has_two_files_in_pretrained_folder():    
            print("Pretrained weights are downloaded. Training tab enabled!\n-------------------------------")       
            with gr.TabItem("Train", visible=False):
                with gr.Row():
                    with gr.Column():
                        exp_dir1 = gr.Textbox(label="Voice Name:", value="My-Voice")
                        sr2 = gr.Radio(
                            label=i18n("目标采样率"),
                            choices=["40k", "48k"],
                            value="40k",
                            interactive=True,
                            visible=False
                        )
                        if_f0_3 = gr.Radio(
                            label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                            choices=[True, False],
                            value=True,
                            interactive=True,
                            visible=False
                        )
                        version19 = gr.Radio(
                            label="RVC version",
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                            visible=False,
                        )
                        np7 = gr.Slider(
                            minimum=0,
                            maximum=config.n_cpu,
                            step=1,
                            label="# of CPUs for data processing (Leave as it is)",
                            value=config.n_cpu,
                            interactive=True,
                            visible=True
                        )
                        trainset_dir4 = gr.Textbox(label="Path to your dataset (audios, not zip):", value="./dataset")
                        easy_uploader = gr.Files(label='OR Drop your audios here. They will be uploaded in your dataset path above.',file_types=['audio'])
                        but1 = gr.Button("1. Process The Dataset", variant="primary")
                        info1 = gr.Textbox(label="Status (wait until it says 'end preprocess'):", value="")
                        easy_uploader.upload(fn=upload_to_dataset, inputs=[easy_uploader, trainset_dir4], outputs=[info1])
                        but1.click(
                            preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1]
                        )
                    with gr.Column():
                        spk_id5 = gr.Slider(
                            minimum=0,
                            maximum=4,
                            step=1,
                            label=i18n("请指定说话人id"),
                            value=0,
                            interactive=True,
                            visible=False
                        )
                        with gr.Accordion('GPU Settings', open=False, visible=False):
                            gpus6 = gr.Textbox(
                                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                value=gpus,
                                interactive=True,
                                visible=False
                            )
                            gpu_info9 = gr.Textbox(label=i18n("显卡信息"), value=gpu_info)
                        f0method8 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢"
                            ),
                            choices=["harvest","crepe", "mangio-crepe", "rmvpe"], # Fork feature: Crepe on f0 extraction for training.
                            value="rmvpe",
                            interactive=True,
                        )
                        
                        extraction_crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label=i18n("crepe_hop_length"),
                            value=128,
                            interactive=True,
                            visible=False,
                        )
                        f0method8.change(fn=whethercrepeornah, inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                        but2 = gr.Button("2. Pitch Extraction", variant="primary")
                        info2 = gr.Textbox(label="Status(Check the Colab Notebook's cell output):", value="", max_lines=8)
                        but2.click(
                                extract_f0_feature,
                                [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, extraction_crepe_hop_length],
                                [info2],
                            )
                    with gr.Row():      
                        with gr.Column():
                            total_epoch11 = gr.Slider(
                                minimum=1,
                                maximum=5000,
                                step=10,
                                label="Total # of training epochs (IF you choose a value too high, your model will sound horribly overtrained.):",
                                value=250,
                                interactive=True,
                            )
                            butstop = gr.Button(
                                "Stop Training",
                                variant='primary',
                                visible=False,
                            )
                            but3 = gr.Button("3. Train Model", variant="primary", visible=True)
                            
                            but3.click(fn=stoptraining, inputs=[gr.Number(value=0, visible=False)], outputs=[but3, butstop])
                            butstop.click(fn=stoptraining, inputs=[gr.Number(value=1, visible=False)], outputs=[butstop, but3])
                            
                            
                            but4 = gr.Button("4.Train Index", variant="primary")
                            info3 = gr.Textbox(label="Status(Check the Colab Notebook's cell output):", value="", max_lines=10)
                            with gr.Accordion("Training Preferences (You can leave these as they are)", open=False):
                                #gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                                with gr.Column():
                                    save_epoch10 = gr.Slider(
                                        minimum=1,
                                        maximum=200,
                                        step=1,
                                        label="Backup every X amount of epochs:",
                                        value=10,
                                        interactive=True,
                                    )
                                    batch_size12 = gr.Slider(
                                        minimum=1,
                                        maximum=40,
                                        step=1,
                                        label="Batch Size (LEAVE IT unless you know what you're doing!):",
                                        value=default_batch_size,
                                        interactive=True,
                                    )
                                    if_save_latest13 = gr.Checkbox(
                                        label="Save only the latest '.ckpt' file to save disk space.",
                                        value=True,
                                        interactive=True,
                                    )
                                    if_cache_gpu17 = gr.Checkbox(
                                        label="Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement.",
                                        value=False,
                                        interactive=True,
                                    )
                                    if_save_every_weights18 = gr.Checkbox(
                                        label="Save a small final model to the 'weights' folder at each save point.",
                                        value=True,
                                        interactive=True,
                                    )
                            zip_model = gr.Button('5. Download Model')
                            zipped_model = gr.Files(label='Your Model and Index file can be downloaded here:')
                            zip_model.click(fn=zip_downloader, inputs=[exp_dir1], outputs=[zipped_model, info3])
                with gr.Group():
                    with gr.Accordion("Base Model Locations:", open=False, visible=False):
                        pretrained_G14 = gr.Textbox(
                            label=i18n("加载预训练底模G路径"),
                            value="pretrained_v2/f0G40k.pth",
                            interactive=True,
                        )
                        pretrained_D15 = gr.Textbox(
                            label=i18n("加载预训练底模D路径"),
                            value="pretrained_v2/f0D40k.pth",
                            interactive=True,
                        )
                        gpus16 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value=gpus,
                            interactive=True,
                        )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, version19],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, pretrained_G14, pretrained_D15],
                    )
                    but5 = gr.Button(i18n("一键训练"), variant="primary", visible=False)
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        [
                            info3,
                            butstop,
                            but3,
                        ],
                    )
                    but4.click(train_index, [exp_dir1, version19], info3)
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                            extraction_crepe_hop_length
                        ],
                        info3,
                    )

        else:
            print(
                "Pretrained weights not downloaded. Disabling training tab.\n"
                "Wondering how to train a voice? Visit here for the RVC model training guide: https://t.ly/RVC_Training_Guide\n"
                "-------------------------------\n"
            )
                
    app.queue(concurrency_count=511, max_size=1022).launch(share=False, quiet=True)
#endregion