import React, { Component } from 'react';
import classnames from 'classnames';
import Collapse from '@material-ui/core/Collapse';
import Icon from '../../../components/uielements/icon/index.js';
import Typography from '../../../components/uielements/typography';
import IconButton from '../../../components/uielements/iconbutton';
import {
  CardContents,
  CardMedias,
  CardAction,
  CardHeaders,
  CardReview,
  Avatar,
} from './card.style';
import CardPic from '../../../images/cards/card-1.jpg';

class RecipeReviewCard extends Component {
  state = { expanded: false };

  handleExpandClick = () => {
    this.setState({ expanded: !this.state.expanded });
  };

  render() {
    const { classes } = this.props;

    return (
      <div>
        <CardReview>
          <CardHeaders
            avatar={<Avatar aria-label="Recipe">R</Avatar>}
            title="Shrimp and Chorizo Paella"
            subheader="September 14, 2016"
          />
          <CardMedias image={CardPic} title="Contemplative Reptile" />
          <CardContents>
            <Typography component="p">
              This impressive paella is a perfect party dish and a fun meal to
              cook together with your guests. Add 1 cup of frozen peas along
              with the mussels, if you like.
            </Typography>
          </CardContents>
          <CardAction disableSpacing>
            <IconButton aria-label="Add to favorites">
              <Icon>favorite</Icon>
            </IconButton>
            <IconButton aria-label="Share">
              <Icon>share</Icon>
            </IconButton>
            <div className={classes.flexGrow} />
            <IconButton
              className={classnames(classes.expand, {
                [classes.expandOpen]: this.state.expanded,
              })}
              onClick={this.handleExpandClick}
              aria-expanded={this.state.expanded}
              aria-label="Show more"
            >
              <Icon>expand_more</Icon>
            </IconButton>
          </CardAction>
          <Collapse
            in={this.state.expanded}
            transitionduration="auto"
            unmountOnExit
          >
            <CardContents>
              <Typography paragraph type="body2">
                Method:
              </Typography>
              <Typography paragraph>
                Heat 1/2 cup of the broth in a pot until simmering, add saffron
                and set aside for 10 minutes.
              </Typography>
              <Typography paragraph>
                Heat oil in a (14- to 16-inch) paella pan or a large, deep
                skillet over medium-high heat. Add chicken, shrimp and chorizo,
                and cook, stirring occasionally until lightly browned, 6 to 8
                minutes. Transfer shrimp to a large plate and set aside, leaving
                chicken and chorizo in the pan. Add pimentón, bay leaves,
                garlic, tomatoes, onion, salt and pepper, and cook, stirring
                often until thickened and fragrant, about 10 minutes. Add
                saffron broth and remaining 4 1/2 cups chicken broth; bring to a
                boil.
              </Typography>
              <Typography paragraph>
                Add rice and stir very gently to distribute. Top with artichokes
                and peppers, and cook without stirring, until most of the liquid
                is absorbed, 15 to 18 minutes. Reduce heat to medium-low, add
                reserved shrimp and mussels, tucking them down into the rice,
                and cook again without stirring, until mussels have opened and
                rice is just tender, 5 to 7 minutes more. (Discard any mussels
                that don’t open.)
              </Typography>
              <Typography>
                Set aside off of the heat to let rest for 10 minutes, and then
                serve.
              </Typography>
            </CardContents>
          </Collapse>
        </CardReview>
      </div>
    );
  }
}

export default RecipeReviewCard;
