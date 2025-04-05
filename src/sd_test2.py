import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
from diffusers import StableDiffusionPipeline, AutoencoderKL
from core.config import settings

import torch
from PIL import Image
import gc

target_sampler_name = "euler_a"

def get_scheduler_factory(sampler_name: str):
    sampler_name = sampler_name.lower()
    sampler_info = settings.SD_SAMPLERS.get(sampler_name)

    if not sampler_info:
        raise ValueError(
            f"未知的採樣器 '{sampler_name}'。\n"
            f"可用選項：{', '.join(settings.SD_SAMPLERS.keys())}"
        )

    SchedulerCls = sampler_info["cls"]
    extra_kwargs = sampler_info.get("kwargs", {})

    return lambda config: SchedulerCls.from_config(config, **extra_kwargs)


# --- 參數設定 ---
#model_dir = "DiscordChatBot/src/Models/anypastelAnythingV45_anypastelAnythingV45"  # 你的主模型資料夾（已轉成 diffusers 格式）
#model_dir = "DiscordChatBot/src/Models/meinaunreal_v5"
#model_dir = "DiscordChatBot/src/Models/meinamix_v12Final"
#------------------------------------------------------------------#
#steps: 40, no need vae
#model_dir = "src/Models/SD/BaseModels/hadrianDelice_deliceV20"
#------------------------------------------------------------------#
#steps: 40, no need vae
model_dir = "src/Models/SD/BaseModels/qchanAnimeMix_v40" 
#------------------------------------------------------------------#
#steps: 20, need vae: ClearVAE_V2
#model_dir = "src/Models/SD/BaseModels/malkmilfmix_v20"
#------------------------------------------------------------------#
#vae_path = "DiscordChatBot/src/Models/converted-orangemixvaeReupload_v10"  # VAE 模型（需要轉換格式）
#vae_path = "src/meinaunreal_v5/kl-f8-anime2.safetensors"  # VAE 模型
#vae_path = "stabilityai/sd-vae-ft-mse"
#vae_path = "DiscordChatBot/src/Models/VAE/converted_vae"
#vae_path = "src/Models/SD/VAE/klF8Anime2VAE_klF8Anime2VAE"
#vae_path = "src/Models/SD/VAE/ClearVAE_V2"
vae_path = ""

embedding_dirs = [
    ("src/Models/SD/Embeddings/easynegative.safetensors", "easynegative"),
    ("src/Models/SD/Embeddings/badhandv4.pt", "badhandv4"),
    ("src/Models/SD/Embeddings/HDA_Ahegao.pt", "HDA_Ahegao")
]

pipe = StableDiffusionPipeline.from_pretrained(
    model_dir, 
    torch_dtype=torch.float32,
    safety_checker=None)  # 禁用 safety checker

sampler_factory = get_scheduler_factory(target_sampler_name)
pipe.scheduler = sampler_factory(pipe.scheduler.config)

# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# # 使用 DPM++ 2M Karras 採樣器
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     use_karras_sigmas=True  # ✅ 開啟 Karras 分布
# )

# 加入vae
#pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to("mps")
if vae_path:
    pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32).to("mps")

pipe = pipe.to("mps")

for path, trigger_word in embedding_dirs:
    pipe.load_textual_inversion(path, token=trigger_word, mean_resizing=False)

pipe.enable_attention_slicing()
#print(pipe.tokenizer.added_tokens_encoder)

lora_weights = { "animetarotV51": 1 , "CivChan": 1}
pipe.load_lora_weights("src/Models/SD/Loras/animetarotV51.safetensors", 
                        adapter_name="animetarotV51")
#CivChan, purple eyes, pink hair
pipe.load_lora_weights("src/Models/SD/Loras/CivChan.safetensors", 
                        adapter_name="CivChan")
#pipe.set_adapters("animetarotV51", scales) 

pipe.set_adapters(list(lora_weights.keys()), list(lora_weights.values()))

#positive_prompt = "masterpiece, best quality, 1girl, solo, light smile, fire, red theme, alternate costume, <lora:animetarotV51:1>"
#positive_prompt = "(masterpiece, best quality), 1girl, solo, long_hair, from top, light smile, panorama, perspective, looking_at_viewer, bangs, skirt, shirt, pink hair, colored inner hair, long_sleeves, bow, ribbon, twintails, hair_bow, heart, pantyhose, frills, shoes, choker, blunt_bangs, black_skirt, pink_eyes, frilled_skirt, pink_bow, platform_footwear, pink_theme, jirai_kei, full body, night, street, skyscraper, neon trim, panorama, perspective, starry sky, pink background, dark, shadow"
# positive_prompt = "(masterpiece, best quality),1girl,adjusting clothes,adjusting headwear,basket,blush,bow,bowtie,breasts,brown eyes,brown hair,cloak,cloud,cloudy sky,crescent moon,dress,fantasy,flower,full body,glowing,glowing flower,hat,holding,holding staff,light particles,lily pad,long hair,looking at viewer,moon,moonlight,mountain,mountainous horizon,night,outdoors,parted lips,pointy ears,pond,sky,small breasts,staff,star (sky),starry sky,very long hair,wading,water lily flower,wicker basket,wind,witch,witch hat"
#positive_prompt = "(masterpiece, best quality), 1girl, solo, long_hair, looking_at_viewer, white hair, red eyes, smile, bangs, flared skirt, pantyhose, shirt, long_sleeves, hat, bow, holding, closed_mouth, flower, frills, hair_flower, petals, bouquet, holding_flower, center_frills, bonnet, holding_bouquet, floral background, flower field, multicolored background, colorful, petals, depth of field, flower field"
#positive_prompt = "(((masterpiece))),(((bestquality))),((ultra-detailed)),(illustration),((anextremelydelicateandbeautiful)),dynamicangle,floating,(beautifuldetailedeyes),(detailedlight) (1girl), solo, floating_hair,glowingeyes,green hair,greeneyes, twintails, halterdress, <lora:atdanStyleLora_lk:0.6>, atdan, green background"
#positive_prompt = "(((masterpiece))),(((bestquality))),1girl, ponytail, white hair, purple eyes, t-shirt, skirt,white thighhighs, flower, petals, light smile,medium breasts, collarbone, depth of field, petals, (illustration:1.1), landscape, background, abstract, mountainous horizon, cloud, sun,"
#positive_prompt = "1girl, renaissance art stylized by Pieter Aertsen, Nurturing, from inside a Bayou, autumn cityscape and Binary star in background, Fall, Anime screencap, Confused, Dramatic spotlight, 800mm lens, surreal design, emotional, surrealism"
#positive_prompt = "1girl, (designed by Junji Ito:0.7) and Moyoco Anno, hip level shot of a Burning , at Sunset, Masterpiece, Depressing, Ambient lighting, Saturated, 'I'm a barbie girl, in a barbie world.', trending on CGSociety, sad"
positive_prompt = "1girl, colorful art stylized by Jim Dine and Maynard Dixon, black platform_footwear, pop art, attractive, landscape of a (The Land of Oz:1.2) , Dramatic, crowded lake, Stars in the sky, Ultra Real, Metalcore, 800mm lens, white stockings, CivChan, purple eyes, pink hair"
#positive_prompt = "masterpiece,best quality,Mechanical prosthetics,cyberpunk,science and technology,human reconstruction,advanced,mechanical promotion,bone remodeling,{body reconstruction},white cathedral (very clean and bright) - many transparent glass windows,white long hair,white stockings,white gloves,messy hair,hair flowing in the wind,high details,extremely delicate and beautiful girl,{{feathers}},small chest,glittering,Solo,white dove,beautiful detail sky,beautiful detail eyes,skirt - gem necklace - gem pendant,bare shoulder,room full of blue crystals (very much),upper body,<lora:animetarotV51:1>,"

# positive_prompt = """
# highly insanely detailed, masterpiece, top quality, best quality, highres, 4k, 8k, RAW photo, (very aesthetic, beautiful and aesthetic), 
# __lazy-wildcards/subject/style-book/tarot_card/prompts__, 
# {0.3::((Ancient Egypt theme, ancient egyptian theme)), ancient egyptian symbol, ancient egyptian hieroglyphic|}, 
# <lora:animeTarotCardArtStyleLora_v31:0.8>, 
# (1girl:1.3), 
# 1other, stockings,
# {__lazy-wildcards/dataset/costume__|}, 
# {0.3::(__lazy-wildcards/subject/costume-ethnicity-body-skin-tone/dark-skinned/prompts__:1.1)|}, 
# {__lazy-wildcards/prompts/hair__|}, 
# {0.8::(__lazy-wildcards/subject/costume-ethnicity-breasts/*/prompts__)|}, 
# {0.3::__lazy-wildcards/subject/costume-ethnicity-body-skin/skindentation/prompts__|}, 
# __lazy-wildcards/dataset/background__, 
# {0.2::__lazy-wildcards/utils/scenery__|}, 
# {0.1::__lazy-wildcards/subject/moment/random/prompts__|}âââ
# """

# positive_prompt = """
# ,, ,, (slim,thin,skinny), masterpiece,4k, best quality,top quality, official art,highest detailed,colorful,distinct_image,, Fuchsia hair,Hairclips,single hair bun,Light Green eyes,pale skin,track suit,wolf ears,(mechanical arms, mechanical legs),medium breasts,jungle,rainforest,ruins,hyper,
# """

# positive_prompt = """
# highly insanely detailed, masterpiece, top quality, best quality, highres, stockings, 4k, 8k, RAW photo, (very aesthetic, beautiful and aesthetic), __lazy-wildcards/subject/style-book/tarot_card/prompts__, {0.3::((Ancient Egypt theme, ancient egyptian theme)), ancient egyptian symbol, ancient egyptian hieroglyphic|}, <lora:animeTarotCardArtStyleLora_v31:1>, (1girl:1.3), 1other, {__lazy-wildcards/dataset/costume__|}, {0.3::(__lazy-wildcards/subject/costume-ethnicity-body-skin-tone/dark-skinned/prompts__:1.1)|}, {__lazy-wildcards/prompts/hair__|}, {0.8::(__lazy-wildcards/subject/costume-ethnicity-breasts/*/prompts__)|}, {0.3::__lazy-wildcards/subject/costume-ethnicity-body-skin/skindentation/prompts__|}, __lazy-wildcards/dataset/background__, {0.2::__lazy-wildcards/utils/scenery__|}, {0.1::__lazy-wildcards/subject/moment/random/prompts__|}âââ
# """

#positive_prompt = "1girl, happy, celebration, white stockings, HDA_Ahegao, nsfw, sculpture art designed by Kathryn Morris Trotter and Jeff Koons, attractive Funko pop, Water color painting of a Golden ratio, (Flower Bulb:1.2) , the Flower Bulb is Enchanting, it is covered in Seashells, Wide view, Illustration, Amusing, concept art, Academism, Autochrome, drawing, womanly, fractal"
negative_prompt = "easynegative, badhandv4, watermark"

seed =  random.randint(0, 2**32 - 1)
#seed = 778722581497242
#seed = 982613921
#seed = 799960941
#seed = 982613921
# seed = 311620705 
#seed = 3872702003
#seed = 2684547527
#seed = 1847947635
#seed = 1349832173

#要產生的圖片數量
num_images = 1
#設定隨機種子
generator = [torch.manual_seed(seed + i) for i in range(num_images)]
guidance_scale = 7
steps = 40
width = 576
height = 1024

imageList = []
for i in range(1):
    generator = torch.manual_seed(seed + i)
    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
        num_images_per_prompt=num_images
    ).images[0]
    imageList.append(image)
    torch.mps.empty_cache()
    gc.collect()

# 儲存圖片
for i, img in enumerate(imageList):
    # 取得模型與 VAE 名稱
    model_name = model_dir.split("/")[-1]
    vae_name = vae_path.split("/")[-1]
    if not vae_name:
        vae_name = None #vae_path.split("/")[-1]
    # lora_name = list(lora_weights.keys())[0]
    # lora_strength = list(lora_weights.values())[0]
    lora_str = "_".join(f"{k}:{v}" for k, v in lora_weights.items())
    folder_path = f"src/results-new/{model_name}/{vae_name}/{target_sampler_name}"

    # embedding_words=""
    # for i in embedding_dirs:
    #     if i[1] in positive_prompt:
    #         embedding_words = f"{embedding_words}_{i[1]}"
    embedding_words = "_".join(i[1] for i in embedding_dirs if i[1] in positive_prompt)
    if embedding_words:
        embedding_words = f"-{embedding_words}"
        
    # 如果資料夾不存在就建立
    os.makedirs(folder_path, exist_ok=True)

    # 組合檔名
    result_path = f"{folder_path}/{lora_str}{embedding_words}-{seed + i}-{steps}.png"
    
    img.save(result_path)
    img.show()