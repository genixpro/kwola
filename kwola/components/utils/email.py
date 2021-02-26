import os
import base64
import traceback
from ...config.logger import getLogger
import pathlib

def sendExperimentResults(config):
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import (
            Mail, Attachment, FileContent, FileName,
            FileType, Disposition, ContentId)

        message = Mail(
            from_email='brad@kwola.io',
            to_emails='brad@kwola.io',
            subject=f'Experiment results: {config.configurationDirectory}',
            html_content=f'These are your results for {config.configurationDirectory}. Please see the attached charts.')

        for file in config.listAllFilesInFolder("charts"):
            data = config.loadKwolaFileData("charts", file)
            encoded = base64.b64encode(data).decode()
            attachment = Attachment()
            attachment.file_content = FileContent(encoded)
            attachment.file_type = FileType('image/png')
            attachment.file_name = FileName(file)
            attachment.disposition = Disposition('attachment')
            attachment.content_id = ContentId(file)
            message.add_attachment(attachment)

        videoFiles = config.listAllFilesInFolder("debug_videos")
        videoFiles = sorted(videoFiles, key=lambda fileName: pathlib.Path(os.path.join(config.configurationDirectory, "debug_videos", fileName)).stat().st_mtime, reverse=True)
        for file in videoFiles[:2]:
            data = config.loadKwolaFileData("debug_videos", file)
            encoded = base64.b64encode(data).decode()
            attachment = Attachment()
            attachment.file_content = FileContent(encoded)
            attachment.file_type = FileType('video/mpeg')
            attachment.file_name = FileName(file)
            attachment.disposition = Disposition('attachment')
            attachment.content_id = ContentId(file)
            message.add_attachment(attachment)

        if 'sendgrid_api_key' in config:
            apiKey = config['sendgrid_api_key']
        elif 'SENDGRID_API_KEY' in os.environ:
            apiKey = os.environ.get('SENDGRID_API_KEY')
        else:
            getLogger().error("There was no API key provided for Sendgrid. Please set sendgrid_api_key within your config.json file.")
            return

        sg = SendGridAPIClient(apiKey)
        response = sg.send(message)
    except Exception:
        getLogger().error(traceback.format_exc())

