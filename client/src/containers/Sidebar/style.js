import styled from 'styled-components';
import { boxShadow } from '../../settings/style-util';
import Drawer from '../../components/uielements/drawers';
import Icons from '../../components/uielements/icon';
import List, {
  ListItem as ListItems,
  ListItemText as ListItemTexts,
} from '../../components/uielements/lists';

const Icon = styled(Icons)``;

const Drawers = styled(Drawer)`
  position: relative;
  height: 100vh;
  z-index: 1301;

  > div:last-child {
    height: 100%;
    width: 300px;
    ${boxShadow('2px 0px 4px 0px rgba(0,0,0,0.3)')};

    .sidebarScrollArea {
      height: calc(100vh - 64px);
    }
  }

  &.f1x3dnAV {
    > div:last-child {
      width: 260px;
      border-right: 0;
      ${boxShadow('none')};
    }
  }
`;

const LogoWrapper = styled.div`
  height: 64px;
  padding: 0 25px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.24);
  ${boxShadow('0px 1px 2px rgba(0, 0, 0, 0.35)')};

  a {
    display: inline-block;
    padding: 10px 0;
    box-sizing: border-box;
    font-size: 21px;
    color: #ffffff;
    font-weight: 300;
    text-decoration: none;
    text-transform: uppercase;
    -webkit-tap-highlight-color: transparent;

    img {
      max-height: 100%;
    }
  }
`;

const Lists = styled(List)`
  padding: 20px 0;

  a {
    -webkit-tap-highlight-color: transparent;
  }
`;

const ListItemIcon = styled(Icon)`
  color: inherit;
  font-size: 18px;
  opacity: 0.75;
  transition: opacity 195ms cubic-bezier(0.4, 0, 0.6, 1);
  width: 22px;
  height: auto;
`;

const ListItemText = styled(ListItemTexts)`
  padding-left: 23px;
  opacity: 0.75;
  transition: opacity 0.2s ease-in-out;

  h3 {
    color: inherit;
    span {
      font-size: 14px;
      color: inherit;
    }
  }

  span {
    font-size: 14px;
    color: inherit;
  }

  &:first-child {
    padding-left: 27px;
  }
`;

const ExpandLessIcon = styled(Icon)`
  font-size: 18px;
  opacity: 0.75;
  transition: opacity 0.2s ease-in-out;
  ${'' /* transition: opacity 195ms cubic-bezier(0.4, 0, 0.6, 1); */};
`;

const ExpandMoreIcon = styled(Icon)`
  font-size: 18px;
  opacity: 0.75;
  transition: opacity 0.2s ease-in-out;
  ${'' /* transition: opacity 195ms cubic-bezier(0.4, 0, 0.6, 1); */};
`;

const ListItem = styled(ListItems)`
  padding: 0 16px 0 25px;
  > a {
    width: 100%;
    text-decoration: none;
  }

  .ListItemWrapper {
    display: flex;
    align-items: center;
    width: 100%;
    padding: 12px 0;
  }

  &:hover {
    background-color: rgba(0, 0, 0, 0.4);
    ${boxShadow('0px 0px 3px 0px rgba(0, 0, 0, 0.2)')};

    ${ListItemText}, ${ListItemIcon}, ${ExpandLessIcon}, ${ExpandMoreIcon} {
      opacity: 1;
    }
  }

  &.expands {
    ${ListItemIcon}, ${ListItemText}, ${ExpandLessIcon}, ${ExpandMoreIcon} {
      opacity: 1;
    }

    + div {
      background-color: rgba(0, 0, 0, 0.25);
      li {
        padding-left: 43px;
      }
    }
  }

  &.selected {
    background-color: rgba(0, 0, 0, 0.4);

    ${ListItemIcon}, ${ListItemText}, ${ExpandLessIcon}, ${ExpandMoreIcon} {
      opacity: 1;
    }
  }
`;

export default Drawers;
export {
  LogoWrapper,
  Lists,
  ListItem,
  ListItemIcon,
  ListItemText,
  Icon,
  ExpandLessIcon,
  ExpandMoreIcon,
};
