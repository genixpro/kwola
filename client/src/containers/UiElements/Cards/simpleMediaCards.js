import React from 'react';
import {
  CardContents,
  CardMedias,
  SimpleCardMedias,
  CardAction,
} from './card.style';
import Button from '../../../components/uielements/button';
import Typography from '../../../components/uielements/typography';
import CardPic from '../../../images/cards/card-1.jpg';

function SimpleMediaCard(props) {
  const { classes } = props;
  return (
    <div>
      <SimpleCardMedias className={classes.card}>
        <CardMedias
          className={classes.media}
          image={CardPic}
          title="Contemplative Reptile"
        />
        <CardContents>
          <Typography variant="h5" component="h2">
            Lizard
          </Typography>
          <Typography component="p">
            Lizards are a widespread group of squamate reptiles, with over 6,000
            species, ranging across all continents except Antarctica
          </Typography>
        </CardContents>
        <CardAction>
          <Button size="small" color="primary">
            Share
          </Button>
          <Button size="small" color="primary">
            Learn More
          </Button>
        </CardAction>
      </SimpleCardMedias>
    </div>
  );
}

export default SimpleMediaCard;
