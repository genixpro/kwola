import styled from 'styled-components';
import Popover from '../../components/uielements/popover';
import IconButton from '../../components/uielements/iconbutton';
import Icons from '../../components/uielements/icon';

const IconButtons = styled(IconButton)`
  width: auto;
  height: auto;
  color: #ffffff;
`;

const SearchIcon = styled(Icons)`
  font-size: 24px;
  color: #ffffff;
`;

const TopbarSearchModal = styled(Popover)`
  z-index: 1401;
  background-color: rgba(0, 0, 0, 0.6);

  > div {
    @media only screen and (max-width: 767px) {
      width: 100%;
      left: 16px !important;
      margin-left: 0;
    }

    .searchContainer {
      > div {
        width: 100%;

        input {
          padding: 16px 20px;
        }

        &:before {
          background-color: transparent;
        }

        &:after {
          background-color: transparent;
        }

        &:hover::before {
          background-color: transparent;
        }
      }
    }
  }
`;

export { IconButtons, SearchIcon };
export default TopbarSearchModal;
