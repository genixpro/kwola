import styled from "styled-components";
import { palette } from "styled-theme";
import Card, {
  CardContent,
  CardMedia,
  CardActions,
  CardHeader
} from "../../../components/uielements/cards";
import Typographys from "../../../components/uielements/typography";
import Avatars from "../../../components/uielements/avatars";
// import Cards from '../../../components/uielements/cards';
import WithDirection from "../../../settings/withDirection";

const Avatar = styled(Avatars)`
  background-color: ${palette("red", 5)};
`;

const Cards = styled(Card)`
  min-width: 275px;
`;

const CardContents = styled(CardContent)`
  flex: 1 0 auto;
`;

const CardAction = styled(CardActions)``;
const Typography = styled(Typographys)``;

const CardHeaders = styled(CardHeader)``;

const CardMedias = styled(CardMedia)`
  width: 151px;
  ${"" /* height: 151px; */} display: inline-flex;
`;

const MediaControlCardsWrapper = styled(Cards)`
  display: inline-flex;

  .details {
    display: flex;
    flex-direction: column;
  }

  .controls {
    display: flex;
    align-items: center;
    padding-left: ${props => (props["data-rtl"] === "rtl" ? "auto" : "8px")};
    padding-right: ${props => (props["data-rtl"] === "rtl" ? "8px" : "auto")};
    padding-bottom: 8px;
  }
`;

const SimpleCards = styled(Cards)`
  display: inline-flex;
  flex-direction: column;

  ${Typography} {
    &p {
      color: ${palette("grey", 8)};
    }

    &.title {
      margin-bottom: 16px;
      font-size: 14px;
      color: ${palette("grey", 6)};
    }

    &.pos {
      margin-bottom: 12px;
      color: ${palette("grey", 6)};
    }
  }
`;

const SimpleCardMedias = styled(Cards)`
  ${CardMedias} {
    width: 100%;
    height: 194px;
  }
`;

const CardReview = styled(Cards)`
  ${CardMedias} {
    width: 100%;
    height: 194px;
  }
`;
const MediaControlCards = WithDirection(MediaControlCardsWrapper);
export {
  Cards,
  CardContents,
  CardAction,
  CardMedias,
  CardHeaders,
  Avatar,
  Typography,
  MediaControlCards,
  SimpleCards,
  SimpleCardMedias,
  CardReview
};
// export default MediaControlCards;
