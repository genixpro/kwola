import React from 'react';
import IconButton from '../../../components/uielements/iconbutton';
import Typography from '../../../components/uielements/typography';
import CardPic from '../../../images/cards/card-1.jpg';
import Icon from '../../../components/uielements/icon/index.js';
import { CardContents, CardMedias, MediaControlCards } from './card.style';

function MediaControlCard(props) {
  const { theme } = props;
  return (
    <div>
      <MediaControlCards>
        <div className="details">
          <CardContents>
            <Typography variant="h5">Live From Space</Typography>
            <Typography variant="subtitle1" color="secondary">
              Mac Miller
            </Typography>
          </CardContents>
          <div className="controls">
            <IconButton aria-label="Previous">
              {theme.direction === 'rtl' ? (
                <Icon>skip_next</Icon>
              ) : (
                <Icon>skip_previous</Icon>
              )}
            </IconButton>
            <IconButton aria-label="Play/pause">
              <Icon>play_arrow</Icon>
            </IconButton>
            <IconButton aria-label="Next">
              {theme.direction === 'rtl' ? (
                <Icon>skip_previous</Icon>
              ) : (
                <Icon>skip_next</Icon>
              )}
            </IconButton>
          </div>
        </div>
        <CardMedias image={CardPic} title="Live from space album cover" />
      </MediaControlCards>
    </div>
  );
}

export default MediaControlCard;
