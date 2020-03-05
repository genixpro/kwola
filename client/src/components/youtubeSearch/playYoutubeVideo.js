import React, { Component } from 'react';
import YouTube from 'react-youtube';
import { Dialog, Icon } from './youtubeSearch.style';

export default class extends Component {
  render() {
    const { selectedVideo, handleCancel } = this.props;
    const ops = { playerVars: { autoplay: 1 } };
    return (
      <Dialog
        open={true}
        onClose={handleCancel}
        className="youtubeVideoModal"
        maxWidth={false}
      >
        <div className="modalContent">
          <button onClick={handleCancel}>
            <Icon>clear</Icon>
          </button>
          <h3>{selectedVideo.snippet.title}</h3>
          <YouTube videoId={selectedVideo.id.videoId} opts={ops} />
        </div>
      </Dialog>
    );
  }
}
