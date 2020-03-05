import styled from 'styled-components';
import { palette } from 'styled-theme';
import Lists, {
  ListItem as ListItems,
  ListItemIcon as ListItemIcons,
  ListItemText as ListItemTexts,
} from '../../uielements/lists';

const List = styled(Lists)`
  padding-bottom: 15px;
`;

const ListItemIcon = styled(ListItemIcons)`
  margin-right: 10px;
`;

const ListItemText = styled(ListItemTexts)`
  h3 {
    font-size: 14px;
    color: ${palette('grey', 8)};
    text-transform: capitalize;
  }
`;

const ListItem = styled(ListItems)`
  cursor: pointer;
  padding: 10px 16px;

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
