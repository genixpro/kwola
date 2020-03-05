import styled from 'styled-components';
import { palette } from 'styled-theme';
import Icons from '../../uielements/icon';
import IconButtons from '../../uielements/iconbutton';
import Lists, {
  ListSubheader as ListSubheaders,
  ListItem as ListItems,
  ListItemText as ListItemTexts,
} from '../../uielements/lists';

const Icon = styled(Icons)`
  font-size: 20px;
  color: ${palette('grey', 7)};
`;

const DoneIcon = styled(Icons)`
  font-size: 22px;
  font-weight: 700;
  color: ${palette('grey', 7)};
`;

const IconButton = styled(IconButtons)`
  width: 30px;
  height: 30px;
  padding: 0;
  margin-right: 5px;

  &:last-child {
    margin-right: 0;
  }
`;

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

const SingleMailActions = styled.div`
  margin-right: -10px;
  display: none;
`;

export {
  Icon,
  DoneIcon,
  IconButton,
  List,
  ListItem,
  ListSubheader,
  ListItemText,
  ListLabel,
};
export default SingleMailActions;
