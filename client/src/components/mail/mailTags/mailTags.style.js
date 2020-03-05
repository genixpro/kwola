import styled from 'styled-components';
import { palette } from 'styled-theme';
import Lists, {
  ListItem as ListItems,
  ListItemIcon as ListItemIcons,
  ListItemText as ListItemTexts,
} from '../../uielements/lists';

const List = styled(Lists)`
  padding-top: 15px;
`;

const ListItemIcon = styled(ListItemIcons)``;

const ListItemText = styled(ListItemTexts)`
  h3 {
    font-size: 14px;
    color: ${palette('grey', 8)};
  }
`;

const ListItem = styled(ListItems)`
  cursor: pointer;

  &:hover {
    background-color: ${palette('grey', 2)};
  }

  &.active {
    background-color: ${palette('grey', 3)};

    ${ListItemText} {
      h3 {
        font-weight: 500;
      }
    }
  }

  .counter {
    border-radius: 50%;
    font-size: 12px;
    background-color: #fff;
    width: 25px;
    height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 500;
    color: ${palette('grey', 8)};
  }
`;

export default List;
export { ListItem, ListItemIcon, ListItemText };
