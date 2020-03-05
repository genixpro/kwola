import styled from 'styled-components';
import { palette } from 'styled-theme';
import { MenuItem as MenuItems } from '../uielements/menus';

const MenuItem = styled(MenuItems)`
  height: auto;
  padding: 10px 16px;

  &:hover {
    background-color: rgba(0, 0, 0, 0.03);
  }

  .userSuggestion {
    display: flex;
    align-items: center;

    .userImg {
      width: 35px;
      height: 35px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border-radius: 50%;
      margin-right: 15px;

      img {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
    }

    .suggetionText {
      font-size: 13px;
      color: ${palette('grey', 8)};
      margin: 0;
    }
  }
`;

export { MenuItem };
