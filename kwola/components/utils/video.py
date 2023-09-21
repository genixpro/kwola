#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#



import subprocess


def getAvailableFfmpegCodecs():
    result = subprocess.run(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = str(result.stdout, 'utf8')
    encoderString = "Encoders:"
    output = output[output.find(encoderString) + len(encoderString):]
    encoderString = "------"
    output = output[output.find(encoderString) + len(encoderString):]

    codecs = [line.strip().split()[1] for line in output.splitlines() if len(line.strip().split()) > 1]

    return codecs


def chooseBestFfmpegVideoCodec(losslessPreferred=False):
    availableVideoCodecs = getAvailableFfmpegCodecs()

    codecPriorityList = [
        'x264',
        "libx264",
        'mpeg4',
        'libwebp',
        'libtheora',
        'libvpx',
        'mjpeg',
        'gif',
    ]

    if losslessPreferred:
        codecPriorityList = ['libx264rgb'] + codecPriorityList

    codec = ''
    for possibleCodec in codecPriorityList:
        if possibleCodec in availableVideoCodecs:
            codec = possibleCodec
            break
    return codec

