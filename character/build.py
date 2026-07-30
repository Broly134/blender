"""Assemblage complet du personnage."""


def assemble(quality=1.0, with_hair=True, with_hat=True, with_glasses=True,
             with_straw=True, with_clothing=True, gaze=(0.012, 0.470, 0.128)):
    from .head import build_head, open_mouth
    from .eyes import cut_eye_openings, build_eyes
    from .ears import build_ears
    from .teeth import build_teeth
    from .skin_attributes import paint
    from .materials import skin_material

    objects = {}

    head = build_head()
    open_mouth(head)
    cut_eye_openings(head)
    paint(head)
    head.data.materials.append(skin_material())
    objects["head"] = head

    ears = build_ears()
    for e in ears:
        paint(e)
        e.data.materials.append(skin_material())
    objects["ears"] = ears

    objects["eyes"] = build_eyes(gaze_target=gaze)
    objects["mouth"] = build_teeth()

    if with_hair:
        from .hair import groom
        groom(head, quality=quality)

    if with_clothing:
        from .clothing import build_clothing
        objects["clothing"] = build_clothing()
    if with_hat:
        from .hat import build_hat
        objects["hat"] = build_hat()
    if with_glasses:
        from .glasses import build_glasses
        objects["glasses"] = build_glasses()
    if with_straw:
        from .straw import build_straw
        objects["straw"] = build_straw()

    return objects


