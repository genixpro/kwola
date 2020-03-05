import React, { Component } from 'react';
import { PropTypes } from 'prop-types';
import SpeechRecognition from 'react-speech-recognition';
import { connectSearchBox } from 'react-instantsearch/connectors';
import { VoiceSearch, MicIcon } from './algoliaComponent.style';

const propTypes = {
  transcript: PropTypes.string,
  resetTranscript: PropTypes.func,
  browserSupportsSpeechRecognition: PropTypes.bool
};
const options = {
  autoStart: false
};
class VoiceRecognition extends Component {
  state = {
    listening: false
  };
  render() {
    const {
      transcript,
      currentRefinement,
      resetTranscript,
      browserSupportsSpeechRecognition,
      startListening,
      stopListening,
      refine
    } = this.props;
    if (!browserSupportsSpeechRecognition) {
      return <div />;
    }
    return (
      <VoiceSearch>
        {!this.state.listening ? (
          <div className="voiceSearchStart">
            <button
              onClick={() => {
                startListening();
                this.setState({ listening: true });
              }}
            >
              <MicIcon>mic</MicIcon>
            </button>
            <span>Start Speaking</span>
          </div>
        ) : (
          <div className="voiceSearchRunning">
            <button
              onClick={() => {
                refine(transcript);
                resetTranscript();
                stopListening();
                this.setState({ listening: false });
              }}
            >
              <MicIcon>pause</MicIcon>
            </button>
            <span>{transcript || currentRefinement}</span>
          </div>
        )}
      </VoiceSearch>
    );
  }
}
VoiceRecognition.propTypes = propTypes;
export default connectSearchBox(SpeechRecognition(options)(VoiceRecognition));
