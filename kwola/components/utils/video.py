#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

