from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import json
import io
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MobileNet Image Classifier", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = None
transform = None
class_names = None

def load_imagenet_classes():
    """Load all 1000 ImageNet class names"""
    return [
        "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead_shark",
        "electric_ray", "stingray", "cock", "hen", "ostrich", "brambling",
        "goldfinch", "house_finch", "junco", "indigo_bunting", "American_robin",
        "bulbul", "jay", "magpie", "chickadee", "water_ouzel", "kite",
        "bald_eagle", "vulture", "great_grey_owl", "European_fire_salamander",
        "common_newt", "eft", "spotted_salamander", "axolotl", "American_bullfrog",
        "tree_frog", "tailed_frog", "loggerhead_sea_turtle", "leatherback_sea_turtle",
        "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "green_iguana",
        "Carolina_anole", "desert_grassland_whiptail_lizard", "agama", "frilled_lizard", "alligator_lizard",
        "Gila_monster", "European_green_lizard", "chameleon", "Komodo_dragon", "Nile_crocodile",
        "American_alligator", "triceratops", "worm_snake", "ring-necked_snake", "eastern_hog-nosed_snake",
        "smooth_green_snake", "kingsnake", "garter_snake", "water_snake", "vine_snake",
        "night_snake", "boa_constrictor", "African_rock_python", "Indian_cobra", "green_mamba",
        "sea_snake", "Saharan_horned_viper", "eastern_diamondback_rattlesnake", "sidewinder_rattlesnake", "trilobite",
        "harvestman", "scorpion", "yellow_garden_spider", "barn_spider", "European_garden_spider",
        "southern_black_widow", "tarantula", "wolf_spider", "tick", "centipede",
        "black_grouse", "ptarmigan", "ruffed_grouse", "prairie_grouse", "peacock",
        "quail", "partridge", "grey_parrot", "macaw", "sulphur-crested_cockatoo",
        "lorikeet", "coucal", "bee_eater", "hornbill", "hummingbird",
        "jacamar", "toucan", "duck", "red-breasted_merganser", "goose",
        "black_swan", "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat", "jellyfish", "sea_anemone", "brain_coral",
        "flatworm", "nematode", "conch", "snail", "slug",
        "sea_slug", "chiton", "chambered_nautilus", "Dungeness_crab", "rock_crab",
        "fiddler_crab", "red_king_crab", "American_lobster", "spiny_lobster", "crayfish",
        "hermit_crab", "isopod", "white_stork", "black_stork", "spoonbill",
        "flamingo", "little_blue_heron", "great_egret", "bittern", "crane",
        "limpkin", "common_gallinule", "American_coot", "bustard", "ruddy_turnstone",
        "dunlin", "common_redshank", "dowitcher", "oystercatcher", "pelican",
        "king_penguin", "albatross", "grey_whale", "killer_whale", "dugong",
        "sea_lion", "Chihuahua", "Japanese_Chin", "Maltese", "Pekingese",
        "Shih_Tzu", "King_Charles_Spaniel", "Papillon", "toy_terrier", "Rhodesian_Ridgeback",
        "Afghan_Hound", "Basset_Hound", "Beagle", "Bloodhound", "Bluetick_Coonhound",
        "Black_and_Tan_Coonhound", "Treeing_Walker_Coonhound", "English_foxhound", "Redbone_Coonhound", "borzoi",
        "Irish_Wolfhound", "Italian_Greyhound", "Whippet", "Ibizan_Hound", "Norwegian_Elkhound",
        "Otterhound", "Saluki", "Scottish_Deerhound", "Weimaraner", "Staffordshire_Bull_Terrier",
        "American_Staffordshire_Terrier", "Bedlington_Terrier", "Border_Terrier", "Kerry_Blue_Terrier", "Irish_Terrier",
        "Norfolk_Terrier", "Norwich_Terrier", "Yorkshire_Terrier", "Wire_Fox_Terrier", "Lakeland_Terrier",
        "Sealyham_Terrier", "Airedale_Terrier", "Cairn_Terrier", "Australian_Terrier", "Dandie_Dinmont_Terrier",
        "Boston_Terrier", "Miniature_Schnauzer", "Giant_Schnauzer", "Standard_Schnauzer", "Scottish_Terrier",
        "Tibetan_Terrier", "Australian_Silky_Terrier", "Soft-coated_Wheaten_Terrier", "West_Highland_White_Terrier", "Lhasa_Apso",
        "Flat-Coated_Retriever", "Curly-coated_Retriever", "Golden_Retriever", "Labrador_Retriever", "Chesapeake_Bay_Retriever",
        "German_Shorthaired_Pointer", "Vizsla", "English_Setter", "Irish_Setter", "Gordon_Setter",
        "Brittany", "Clumber_Spaniel", "English_Springer_Spaniel", "Welsh_Springer_Spaniel", "Cocker_Spaniel",
        "Sussex_Spaniel", "Irish_Water_Spaniel", "Kuvasz", "Schipperke", "Groenendael",
        "Malinois", "Briard", "Australian_Kelpie", "Komondor", "Old_English_Sheepdog",
        "Shetland_Sheepdog", "collie", "Border_Collie", "Bouvier_des_Flandres", "Rottweiler",
        "German_Shepherd_Dog", "Dobermann", "Miniature_Pinscher", "Greater_Swiss_Mountain_Dog", "Bernese_Mountain_Dog",
        "Appenzeller_Sennenhund", "Entlebucher_Sennenhund", "Boxer", "Bullmastiff", "Tibetan_Mastiff",
        "French_Bulldog", "Great_Dane", "St._Bernard", "husky", "Alaskan_Malamute",
        "Siberian_Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug",
        "Leonberger", "Newfoundland_dog", "Great_Pyrenees", "Samoyed", "Pomeranian",
        "Chow_Chow", "Keeshond", "brussels_griffon", "Pembroke_Welsh_Corgi", "Cardigan_Welsh_Corgi",
        "Toy_Poodle", "Miniature_Poodle", "Standard_Poodle", "Mexican_hairless_dog", "grey_wolf",
        "Alaskan_tundra_wolf", "red_wolf", "coyote", "dingo", "dhole",
        "African_wild_dog", "hyena", "red_fox", "kit_fox", "Arctic_fox",
        "grey_fox", "tabby_cat", "tiger_cat", "Persian_cat", "Siamese_cat",
        "Egyptian_Mau", "cougar", "lynx", "leopard", "snow_leopard",
        "jaguar", "lion", "tiger", "cheetah", "brown_bear",
        "American_black_bear", "polar_bear", "sloth_bear", "mongoose", "meerkat",
        "tiger_beetle", "ladybug", "ground_beetle", "longhorn_beetle", "leaf_beetle",
        "dung_beetle", "rhinoceros_beetle", "weevil", "fly", "bee",
        "ant", "grasshopper", "cricket", "stick_insect", "cockroach",
        "praying_mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
        "damselfly", "red_admiral", "ringlet", "monarch_butterfly", "small_white",
        "sulphur_butterfly", "gossamer-winged_butterfly", "starfish", "sea_urchin", "sea_cucumber",
        "cottontail_rabbit", "hare", "Angora_rabbit", "hamster", "porcupine",
        "fox_squirrel", "marmot", "beaver", "guinea_pig", "common_sorrel",
        "zebra", "pig", "wild_boar", "warthog", "hippopotamus",
        "ox", "water_buffalo", "bison", "ram", "bighorn_sheep",
        "Alpine_ibex", "hartebeest", "impala", "gazelle", "dromedary",
        "llama", "weasel", "mink", "European_polecat", "black-footed_ferret",
        "otter", "skunk", "badger", "armadillo", "three-toed_sloth",
        "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
        "guenon", "patas_monkey", "baboon", "macaque", "langur",
        "black-and-white_colobus_monkey", "proboscis_monkey", "marmoset", "white-headed_capuchin", "howler_monkey",
        "titi_monkey", "Geoffroy's_spider_monkey", "common_squirrel_monkey", "ring-tailed_lemur", "indri",
        "Asian_elephant", "African_bush_elephant", "red_panda", "giant_panda", "snoek",
        "eel", "coho_salmon", "rock_beauty", "anemone_fish", "sturgeon",
        "gar", "lionfish", "pufferfish", "abacus", "abaya",
        "academic_gown", "accordion", "acoustic_guitar", "aircraft_carrier", "airliner",
        "airship", "altar", "ambulance", "amphibian", "analog_clock",
        "apiary", "apron", "trash_can", "assault_rifle", "backpack",
        "bakery", "balance_beam", "balloon", "ballpoint_pen", "Band-Aid",
        "banjo", "baluster", "barbell", "barber_chair", "barbershop",
        "barn", "barometer", "barrel", "wheelbarrow", "baseball",
        "basketball", "bassinet", "bassoon", "swimming_cap", "bath_towel",
        "bathtub", "station_wagon", "lighthouse", "beaker", "military_cap",
        "beer_bottle", "beer_glass", "bell_tower", "baby_bib", "tandem_bicycle",
        "bikini", "ring_binder", "binoculars", "birdhouse", "boathouse",
        "bobsleigh", "bolo_tie", "poke_bonnet", "bookcase", "bookstore",
        "bottle_cap", "hunting_bow", "bow_tie", "brass_memorial_plaque", "bra",
        "breakwater", "breastplate", "broom", "bucket", "buckle",
        "bulletproof_vest", "high-speed_train", "butcher_shop", "taxicab", "cauldron",
        "candle", "cannon", "canoe", "can_opener", "cardigan", "car_mirror",
        "carousel", "tool_kit", "cardboard_box", "car_wheel", "automated_teller_machine",
        "cassette", "cassette_player", "castle", "catamaran", "CD_player",
        "cello", "mobile_phone", "chain", "chain-link_fence", "chain_mail",
        "chainsaw", "storage_chest", "chiffonier", "bell_tower", "china_cabinet",
        "Christmas_stocking", "church", "movie_theater", "cleaver", "cliff_dwelling",
        "cloak", "clogs", "cocktail_shaker", "coffee_mug", "coffeemaker",
        "spiral", "combination_lock", "computer_keyboard", "candy_store", "container_ship",
        "convertible", "corkscrew", "cornet", "cowboy_boot", "cowboy_hat",
        "cradle", "construction_crane", "crash_helmet", "crate", "infant_bed",
        "Crock_Pot", "croquet_ball", "crutch", "cuirass", "dam",
        "desk", "desktop_computer", "rotary_dial_telephone", "diaper", "digital_clock",
        "digital_watch", "dining_table", "dishcloth", "dishwasher", "disc_brake",
        "dock", "dog_sled", "dome", "doormat", "drilling_rig",
        "drum", "drumstick", "dumbbell", "Dutch_oven", "electric_fan",
        "electric_guitar", "electric_locomotive", "entertainment_center", "envelope", "espresso_machine",
        "face_powder", "feather_boa", "filing_cabinet", "fireboat", "fire_truck",
        "fire_screen", "flagpole", "flute", "folding_chair", "football_helmet",
        "forklift", "fountain", "fountain_pen", "four-poster_bed", "freight_car",
        "French_horn", "frying_pan", "fur_coat", "garbage_truck", "gas_mask",
        "gas_pump", "goblet", "go-kart", "golf_ball", "golf_cart",
        "gondola", "gong", "gown", "grand_piano", "greenhouse",
        "grille", "grocery_store", "guillotine", "barrette", "hair_spray",
        "half-track", "hammer", "hamper", "hair_dryer", "hand-held_computer",
        "handkerchief", "hard_disk_drive", "harmonica", "harp", "combine_harvester",
        "hatchet", "holster", "home_theater", "honeycomb", "hook",
        "hoop_skirt", "gymnastics_horizontal_bar", "horse-drawn_vehicle", "hourglass", "iPod",
        "clothes_iron", "carved_pumpkin", "jeans", "jeep", "T-shirt",
        "jigsaw_puzzle", "rickshaw", "joystick", "kimono", "knee_pad",
        "knot", "lab_coat", "ladle", "lampshade", "laptop_computer",
        "lawn_mower", "lens_cap", "letter_opener", "library", "lifeboat",
        "lighter", "limousine", "ocean_liner", "lipstick", "slip-on_shoe",
        "lotion", "music_speaker", "loupe_magnifying_glass", "sawmill", "magnetic_compass",
        "messenger_bag", "mailbox", "maillot", "maillot_tank_suit", "manhole_cover",
        "maraca", "marimba", "mask", "matchstick", "maypole",
        "maze", "measuring_cup", "medicine_chest", "megalith", "microphone",
        "microwave_oven", "military_uniform", "milk_can", "minibus", "miniskirt",
        "minivan", "missile", "mitten", "mixing_bowl", "mobile_home",
        "Model_T", "modem", "monastery", "monitor", "moped",
        "mortar_and_pestle", "graduation_cap", "mosque", "mosquito_net", "vespa",
        "mountain_bike", "tent", "computer_mouse", "mousetrap", "moving_van",
        "muzzle", "metal_nail", "neck_brace", "necklace", "baby_pacifier",
        "notebook_computer", "obelisk", "oboe", "ocarina", "odometer",
        "oil_filter", "pipe_organ", "oscilloscope", "overskirt", "bullock_cart",
        "oxygen_mask", "product_packet", "paddle", "paddle_wheel", "padlock",
        "paintbrush", "pajamas", "palace", "pan_flute", "paper_towel",
        "parachute", "parallel_bars", "park_bench", "parking_meter", "railroad_car",
        "patio", "payphone", "pedestal", "pencil_case", "pencil_sharpener",
        "perfume", "Petri_dish", "photocopier", "plectrum", "Pickelhaube",
        "picket_fence", "pickup_truck", "pier", "piggy_bank", "pill_bottle",
        "pillow", "ping-pong_ball", "pinwheel", "pirate_ship", "drink_pitcher",
        "block_plane", "planetarium", "plastic_bag", "plate_rack", "farm_plow",
        "plunger", "Polaroid_camera", "pole", "police_van", "poncho",
        "billiard_table", "soda_bottle", "plant_pot", "potter's_wheel", "power_drill",
        "prayer_rug", "printer", "prison", "missile", "projector",
        "hockey_puck", "punching_bag", "purse", "quill", "quilt",
        "race_car", "racket", "radiator", "radio", "radio_telescope",
        "rain_barrel", "recreational_vehicle", "fishing_casting_reel", "reflex_camera", "refrigerator",
        "remote_control", "restaurant", "revolver", "rifle", "rocking_chair",
        "rotisserie", "eraser", "rugby_ball", "ruler_measuring_stick", "sneaker",
        "safe", "safety_pin", "salt_shaker", "sandal", "sarong",
        "saxophone", "scabbard", "weighing_scale", "school_bus", "schooner",
        "scoreboard", "CRT_screen", "screw", "screwdriver", "seat_belt",
        "sewing_machine", "shield", "shoe_store", "shoji_screen", "shopping_basket",
        "shopping_cart", "shovel", "shower_cap", "shower_curtain", "ski",
        "balaclava_ski_mask", "sleeping_bag", "slide_rule", "sliding_door", "slot_machine",
        "snorkel", "snowmobile", "snowplow", "soap_dispenser", "soccer_ball",
        "sock", "solar_thermal_collector", "sombrero", "soup_bowl", "keyboard_space_bar",
        "space_heater", "space_shuttle", "spatula", "motorboat", "spider_web",
        "spindle", "sports_car", "spotlight", "stage", "steam_locomotive",
        "through_arch_bridge", "steel_drum", "stethoscope", "scarf", "stone_wall",
        "stopwatch", "stove", "strainer", "tram", "stretcher",
        "couch", "stupa", "submarine", "suit", "sundial",
        "sunglasses", "sunglasses", "sunscreen", "suspension_bridge", "mop",
        "sweatshirt", "swim_trunks", "swing", "electrical_switch", "syringe",
        "table_lamp", "tank", "tape_player", "teapot", "teddy_bear",
        "television", "tennis_ball", "thatched_roof", "front_curtain", "thimble",
        "threshing_machine", "throne", "tile_roof", "toaster", "tobacco_shop",
        "toilet_seat", "torch", "totem_pole", "tow_truck", "toy_store",
        "tractor", "semi-trailer_truck", "tray", "trench_coat", "tricycle",
        "trimaran", "tripod", "triumphal_arch", "trolleybus", "trombone",
        "hot_tub", "turnstile", "typewriter_keyboard", "umbrella", "unicycle",
        "upright_piano", "vacuum_cleaner", "vase", "vault", "velvet",
        "vending_machine", "vestment", "viaduct", "violin", "volleyball",
        "waffle_iron", "wall_clock", "wallet", "wardrobe", "military_aircraft",
        "sink", "washing_machine", "water_bottle", "water_jug", "water_tower",
        "whiskey_jug", "whistle", "wig", "window_screen", "window_shade",
        "Windsor_tie", "wine_bottle", "airplane_wing", "wok", "wooden_spoon",
        "wool", "split-rail_fence", "shipwreck", "sailboat", "yurt",
        "website", "comic_book", "crossword", "traffic_or_street_sign", "traffic_light",
        "dust_jacket", "menu", "plate", "guacamole", "consomme",
        "hot_pot", "trifle", "ice_cream", "popsicle", "baguette",
        "bagel", "pretzel", "cheeseburger", "hot_dog", "mashed_potato",
        "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti_squash",
        "acorn_squash", "butternut_squash", "cucumber", "artichoke", "bell_pepper",
        "cardoon", "mushroom", "Granny_Smith_apple", "strawberry", "orange",
        "lemon", "fig", "pineapple", "banana", "jackfruit",
        "cherimoya", "pomegranate", "hay", "carbonara", "chocolate_syrup",
        "dough", "meatloaf", "pizza", "pot_pie", "burrito",
        "red_wine", "espresso", "tea_cup", "eggnog", "mountain",
        "bubble", "cliff", "coral_reef", "geyser", "lakeshore",
        "promontory", "sandbar", "beach", "valley", "volcano",
        "baseball_player", "bridegroom", "scuba_diver", "rapeseed", "daisy",
        "yellow_lady's_slipper", "corn", "acorn", "rose_hip", "horse_chestnut_seed",
        "coral_fungus", "agaric", "gyromitra", "stinkhorn_mushroom", "earth_star_fungus",
        "hen_of_the_woods_mushroom", "bolete", "corn_cob", "toilet_paper"
    ]

def load_model():
    global model, transform, class_names
    
    logger.info("Loading MobileNetV2 model...")
    
    model = mobilenet_v2(pretrained=True)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    class_names = load_imagenet_classes()
    
    logger.info(f"Model loaded successfully! Ready to classify {len(class_names)} classes.")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "total_classes": len(class_names) if class_names else 0
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict image class"""
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be an image"}
            )
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(5):
            idx = top5_indices[i].item()
            prob = top5_prob[i].item()
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            predictions.append({
                "class": class_name.replace("_", " ").title(),
                "confidence": round(prob * 100, 2)
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "filename": file.filename,
            "image_size": f"{image.size[0]}x{image.size[1]}"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)