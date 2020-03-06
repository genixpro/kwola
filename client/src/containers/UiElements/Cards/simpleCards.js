import React from 'react';
import {
  SimpleCards,
  CardContents,
  CardAction,
  Typography,
} from './card.style';
import Button from '../../../components/uielements/button';
// import Typography from "../../../components/uielements/typography";

function SimpleCard(props) {
  const { classes } = props;
  const bull = <span className={classes.bullet}>•</span>;

  return (
    <div>
      <SimpleCards>
        <CardContents>
          <Typography variant="body2" className="title">
            Word of the Day
          </Typography>
          <Typography variant="h5" component="h2">
            be{bull}nev{bull}o{bull}lent
          </Typography>
          <Typography variant="body2" className="pos">
            adjective
          </Typography>
          <Typography component="p">
            well meaning and kindly.
            <br />
            {'"a benevolent smile"'}
          </Typography>
        </CardContents>
        <CardAction>
          <Button size="small">Learn More</Button>
        </CardAction>
      </SimpleCards>
    </div>
  );
}

export default SimpleCard;
