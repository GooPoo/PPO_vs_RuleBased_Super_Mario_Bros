from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

# code for locating objects on the screen in super mario bros
# by Benjamin Lee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID = False
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT = 240
SCREEN_WIDTH = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": [
            "marioA.png",
            "marioB.png",
            "marioC.png",
            "marioD.png",
            "marioE.png",
            "marioF.png",
            "marioG.png",
        ],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png", "block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.
        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    },
}


def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0] * image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None  # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions


def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results


def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results


# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ",  # sky blue colour
    (0, 0, 0): " ",  # black
    (252, 252, 252): "'",  # white / cloud colour
    (248, 56, 0): "M",  # red / mario colour
    (228, 92, 16): "%",  # brown enemy / block colour
}
unused_letters = sorted(
    set(string.ascii_uppercase) - set(colour_map.values()), reverse=True
)
DEFAULT_LETTER = "?"


def _get_colour(colour):  # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]

    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER


def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x + i, y)] = name_str[i % len(name_str)]
                pixels[(x + i, y + height - 1)] = name_str[
                    (i + height - 1) % len(name_str)
                ]
            for i in range(1, height - 1):
                pixels[(x, y + i)] = name_str[i % len(name_str)]
                pixels[(x + width - 1, y + i)] = name_str[
                    (i + width - 1) % len(name_str)
                ]

    # print the screen to terminal
    print("-" * SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))


################################################################################
# LOCATING OBJECTS


def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break

    #      [((x,y), (width,height))]
    return [(loc, locations[loc]) for loc in locations]


def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(
        screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask
    )
    upper_locs = list(zip(*np.where(upper_results >= threshold)))

    # stop early if there are no pipes
    if not upper_locs:
        return []

    # find the lower part of the pipe
    lower_results = cv.matchTemplate(
        screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask
    )
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y + h, x + 2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations


def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue

            # find locations of objects
            results = _locate_object(
                screen, category_templates[object_name], stop_early
            )
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations


################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION


def make_action(screen, info, step, env, action):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # You probably don't want to print everything I am printing when you run
    # your code, because printing slows things down, and it puts a LOT of
    # information in your terminal.

    # Printing the whole grid is slow, so I am only printing it occasionally,
    # and I'm only printing it for debug purposes, to see if I'm locating objects
    # correctly.
    if PRINT_GRID and step % 100 == 0:
        print_grid(screen, object_locations)
        # If printing the grid doesn't display in an understandable way, change
        # the settings of your terminal (or anaconda prompt) to have a smaller
        # font size, so that everything fits on the screen. Also, use a large
        # terminal window / whole screen.

        # object_locations contains the locations of all the objects we found
        print(object_locations)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]

    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]

    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
            print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")

        # Or you could do it this way:
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
            print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
        print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in object_locations are screen coordinates.
        # Top left corner of screen is (0, 0), top right corner is (255, 0).
        # Here's how you can get Mario's screen coordinates:
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            print(
                "Mario's location on screen:",
                mario_x,
                mario_y,
                f"({object_name} mario)",
            )

        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        print(
            "Mario's location in world:",
            mario_world_x,
            mario_world_y,
            f"({mario_status} mario)",
        )

    # TODO: Write code for a strategy, such as a rule based agent.

    # Choose an action from the list of available actions.
    # For example, action = 0 means do nothing
    #              action = 1 means press 'right' button
    #              action = 2 means press 'right' and 'A' buttons at the same time
    #              action = 3 means press 'right' button & sprint
    #              action = 4 means press 'right' and 'A' & 'B'
    #              action = 5 means press 'A' button
    #              action = 6 means press 'left' and 'A'
    #              action = 7 means press 'left' and 'B'
    #              action = 8 means press 'left' and 'A' & 'B'
    #              action = 9 means press 'down'
    #              action = 10 means press 'up'

    global in_jump_mode
    global jump_hold_counter
    if str(info["world"]) == "1" and str(info["stage"]) == "1":
        if int(info["y_pos"]) > 250:
            action = 1
            return action
        if int(info["x_pos"]) > 280 and int(info["x_pos"] < 390):
            action = 1
            return action
        if int(info["x_pos"]) > 1530 and int(info["x_pos"] < 1560):
            action = 1
            return action

    if in_jump_mode:
        jump_hold_counter += 1
        if jump_hold_counter < JUMP_HOLD_LIMIT:
            action = 4
            return action
        else:
            in_jump_mode = False
            jump_hold_counter = 0
            action = 3
            return action

    if step % 5 == 0:
        location, dimensions, object_name = mario_locations[0]
        mario_x, mario_y = location # Screen coordinates
        enemy_nearby = any(mario_x + 40 > enemy[0][0] > mario_x for enemy in enemy_locations)
        skip = False
        if int(info["x_pos"]) > 1850 and int(info["x_pos"] < 2000):
            skip = True

        if enemy_nearby and not skip:
            for enemy in enemy_locations:
                enemy_location, _, _ = enemy
                enemy_x, enemy_y = enemy_location
                if enemy_y + 10 < mario_y:
                    action = 1
                    return action
            in_jump_mode = True
            jump_hold_counter = 0
            action = 2
            return action

        pipes_ahead = [loc for loc, _, obj_name in object_locations["block"] if obj_name == "pipe" and loc[0] > mario_x]

        if pipes_ahead and not enemy_nearby:
            nearest_pipe_x, nearest_pipe_y = pipes_ahead[0] 
            
            HORIZONTAL_THRESHOLD = 30
            
            if nearest_pipe_x - mario_x <= HORIZONTAL_THRESHOLD and mario_y >= nearest_pipe_y-10:
                in_jump_mode = True
                jump_hold_counter = 0
                action = 2
                return action

        action_from_pixels = detectPixels(obs, mario_x, mario_y)
        if action_from_pixels != 1: 
            return action_from_pixels
        action = 1
        return action
    else:
        return action

def detectPixels(obs, mario_x, enemy_locations):
    SKY_COLOR = (104, 136, 252)
    ENEMY_OR_BLOCK_COLOR = (228, 92, 16)

    HORIZON_CHECK = 20

    checkpoint = 200 
    
    global in_jump_mode
    global jump_hold_counter

    if in_jump_mode:
        jump_hold_counter += 1
        if jump_hold_counter < JUMP_HOLD_LIMIT:
            action = 4
            return action
        else:
            in_jump_mode = False
            jump_hold_counter = 0
            action = 3
            return action

    start_x = max(0, mario_x)
    end_x = min(SCREEN_WIDTH, mario_x + HORIZON_CHECK)
    
    for x in range(start_x, end_x):
        color_at_position = obs[checkpoint, x]
        sky_at_position = obs[checkpoint+15, x+5]
    
        if np.array_equal(color_at_position, ENEMY_OR_BLOCK_COLOR):
            in_jump_mode = True
            jump_hold_counter = 0
            return 2
        if np.array_equal(sky_at_position, SKY_COLOR):
            in_jump_mode = True
            jump_hold_counter = 0
            return 2
    return 1


################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

JUMP_HOLD_LIMIT = 30
in_jump_mode = False
jump_hold_counter = 0

obs = None
done = True
total_reward = 0
total_actions = 0
env.reset()
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    total_actions += 1
    if done:
        print("Furthest x position:"+str(info["x_pos"]))
        print("Total reward:"+str(total_reward))
        print("Total actions:"+str(total_actions))
        env.reset()
        total_reward = 0
        total_actions = 0
env.close()

