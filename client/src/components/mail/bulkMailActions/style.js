import styled from 'styled-components';
import { palette } from 'styled-theme';
import Icons from '../../uielements/icon';
import IconButtons from '../../uielements/iconbutton';
import Checkboxs from '../../uielements/checkbox';
import Popovers from '../../uielements/popover';
import Lists, {
  ListSubheader as ListSubheaders,
  ListItem as ListItems,
  ListItemText as ListItemTexts,
} from '../../uielements/lists';

const IconButton = styled(IconButtons)`
  width: 35px;
  height: 35px;
  padding: 0;
`;

const Icon = styled(Icons)`
  color: #ffffff;

  @media only screen and (max-width: 1099px) {
    font-size: 22px;
  }
`;

const DoneIcon = styled(Icons)`
  font-size: 24px;
  color: #ffffff;

  @media only screen and (max-width: 1099px) {
    font-size: 22px;
  }
`;

const Checkbox = styled(Checkboxs)`
  color: #fff;
`;

const Popover = styled(Popovers)``;

const List = styled(Lists)`
  max-width: 360px;
  width: 100%;

  &.dropdownList {
    padding-bottom: 0;
  }
`;

const ListSubheader = styled(ListSubheaders)`
  height: 35px;
  display: flex;
  width: 100%;
  align-items: center;
  font-size: 12px;
  color: ${palette('grey', 7)};
  font-weight: 500;
  padding-left: 22px;
`;

const ListItem = styled(ListItems)`
  padding: 8px 20px;

  ${Icon} {
    font-size: 19px;
  }
`;

const ListItemText = styled(ListItemTexts)`
  h3 {
    color: ${palette('grey', 6)};
    font-size: 13px;
    text-transform: capitalize;
    font-weight: 500;
  }
`;

const ListLabel = styled.h3`
  font-size: 15px;
  font-weight: 500;
  color: ${palette('grey', 8)};
  padding: 6px 20px 14px;
  margin: 0;
`;

const BulkMailActionWrapper = styled.div`
  width: 100%;
  display: flex;
  align-items: center;
  z-index: 2;
  padding: 0 20px;
  padding-left: 18px;

  @media only screen and (max-width: 1099px) {
    padding: 0 10px;
  }
`;

const LeftPart = styled.div`
  display: flex;
  align-items: center;

  .selectedNum {
    font-size: 16px;
    color: #ffffff;
    margin-left: 5px;
  }
`;

const RightPart = styled.div`
  display: flex;
  align-items: center;
  margin-left: auto;
`;

export {
  LeftPart,
  RightPart,
  Popover,
  Checkbox,
  Icon,
  DoneIcon,
  IconButton,
  List,
  ListItem,
  ListSubheader,
  ListItemText,
  ListLabel,
};
export default BulkMailActionWrapper;
