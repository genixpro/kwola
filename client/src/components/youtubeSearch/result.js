import React, { Component } from 'react';
import Loader from '../utility/Loader';
import HelperText from '../utility/helper-text';
import Button from '../uielements/button';
import PlayYoutubeVideo from './playYoutubeVideo';
import { DemoWrapper } from '../utility/papersheet';
import {
  YoutubeSearchListStyleWrapper,
  YoutubeSearchStyleWrapper,
  PapersheetStyle,
} from './youtubeSearch.style';

function SearchList(result, handleSelectedVideo) {
  return (
    <YoutubeSearchListStyleWrapper className="YoutubeResultList">
      {result.map(item => {
        const {
          publishedAt,
          title,
          description,
          channelTitle,
          thumbnails,
          channelId,
        } = item.snippet;
        const id = item.id.videoId;
        const updateDate = new Date(publishedAt).toDateString();
        const onClick = event => {
          if (event) {
            event.preventDefault();
          }
          handleSelectedVideo(item);
        };
        const onChannelClick = event => {
          event.preventDefault();
          event.stopPropagation();
          window.open(`https://www.youtube.com/channel/${channelId}`, '_blank');
        };
        return (
          <PapersheetStyle
            key={id}
            className="SingleVideoResult"
            onClick={onClick}
          >
            <div className="videoThumb">
              <img alt="#" src={thumbnails.default.url} />
            </div>

            <div className="videoDescription">
              <h3 className="videoName">
                <a href="#!">{`${title} `}</a>
              </h3>

              <div className="ChannelNameAndDate">
                <a href="#!" onClick={onChannelClick} className="channelTitle">
                  {`${channelTitle} `}
                </a>
                <span className="uploadDate">{updateDate}</span>
              </div>

              {description ? <p>{description}</p> : ''}
            </div>
          </PapersheetStyle>
        );
      })}
    </YoutubeSearchListStyleWrapper>
  );
}
class YoutubeResult extends Component {
  state = {
    selectedVideo: null,
  };
  handleCancel = () => {
    this.handleSelectedVideo(null);
  };
  handleSelectedVideo = selectedVideo => {
    this.setState({ selectedVideo });
  };
  render() {
    const { YoutubeSearch, onPageChange } = this.props;
    const { selectedVideo } = this.state;
    const {
      searcText,
      result,
      loading,
      error,
      nextPageToken,
      prevPageToken,
      total_count,
    } = YoutubeSearch;
    if (!searcText) {
      return <div />;
    }
    if (loading) {
      return (
        <DemoWrapper>
          <Loader />
        </DemoWrapper>
      );
    }
    if (error || !total_count) {
      return <HelperText text="THERE ARE SOME ERRORS" />;
    }
    if (result.length === 0) {
      return <HelperText text="No Result Found" />;
    }
    return (
      <YoutubeSearchStyleWrapper className="YoutubeSearchResult">
        <p className="TotalResultFind">
          <span>{`${total_count}`} videos found</span>
        </p>
        {selectedVideo ? (
          <PlayYoutubeVideo
            selectedVideo={this.state.selectedVideo}
            handleCancel={this.handleCancel}
          />
        ) : (
          ''
        )}
        <DemoWrapper>
          {SearchList(result, this.handleSelectedVideo)}
        </DemoWrapper>

        <div className="youtubeSearchPagination">
          <Button
            disabled={prevPageToken ? false : true}
            onClick={() => onPageChange(searcText, prevPageToken)}
          >
            Previous
          </Button>

          {nextPageToken ? (
            <Button onClick={() => onPageChange(searcText, nextPageToken)}>
              Next
            </Button>
          ) : (
            ''
          )}
        </div>
      </YoutubeSearchStyleWrapper>
    );
  }
}
export default YoutubeResult;
