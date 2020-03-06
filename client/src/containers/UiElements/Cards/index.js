import React, { Component } from "react";
import { withStyles } from '@material-ui/core/styles';
import IntlMessages from "../../../components/utility/intlMessages";
import LayoutWrapper from "../../../components/utility/layoutWrapper";
import { Row, HalfColumn } from "../../../components/utility/rowColumn";

import Papersheet, {
  DemoWrapper
} from "../../../components/utility/papersheet";
import RecipeReviewCard from "./cardReview";
import SimpleCard from "./simpleCards";
import SimpleMediaCard from "./simpleMediaCards";
import MediaControlCard from "./mediaControlCards";

const styles = theme => ({
  bullet: {
    display: "inline-block",
    margin: "0 2px",
    transform: "scale(0.8)"
  },
  expand: {
    transform: "rotate(0deg)",
    transition: theme.transitions.create("transform", {
      duration: theme.transitions.duration.shortest
    })
  },
  expandOpen: {
    transform: "rotate(180deg)"
  },
  flexGrow: {
    flex: "1 1 auto"
  }
});

class CardsExample extends Component {
  render() {
    const { props } = this;
    return (
      <LayoutWrapper>
        <Row>
          <HalfColumn>
            <Papersheet
              title={<IntlMessages id="uielements.cards.sampleCard" />}
              codeBlock="UiElements/Cards/simpleCards.js"
              stretched
            >
              <p>
                Although cards can support multiple actions, UI controls, and an
                overflow menu, use restraint and remember that cards are entry
                points to more complex and detailed information.
              </p>
              <DemoWrapper>
                <SimpleCard {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Media"
              codeBlock="UiElements/Cards/simpleMediaCards.js"
              stretched
            >
              <p>Example of a card using an image to reinforce the content.</p>

              <DemoWrapper>
                <SimpleMediaCard {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
        <Row>
          <HalfColumn>
            <Papersheet
              title="UI Controls"
              codeBlock="UiElements/Cards/MediaControlCards.js"
              stretched
            >
              <p>
                Supplemental actions within the card are explicitly called out
                using icons, text, and UI controls, typically placed at the
                bottom of the card. <br />
                <br /> Here's an example of a media control card.
              </p>

              <DemoWrapper>
                <MediaControlCard {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>

          <HalfColumn>
            <Papersheet
              title="Badges"
              codeBlock="UiElements/Cards/cardReview.js"
              stretched
            >
              <p>
                Although cards can support multiple actions, UI controls, and an
                overflow menu, use restraint and remember that cards are entry
                points to more complex and detailed information.
              </p>

              <DemoWrapper>
                <RecipeReviewCard {...props} />
              </DemoWrapper>
            </Papersheet>
          </HalfColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}
export default withStyles(styles, { withTheme: true })(CardsExample);
