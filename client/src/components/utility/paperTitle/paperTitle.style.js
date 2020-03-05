import styled from 'styled-components';
import { palette } from 'styled-theme';

const PageTitle = styled.div`
  width: auto;
  padding: 25px 30px;
  border-bottom: 1px solid ${palette('grey', 3)};

  &.single {
    width: 100%;
    padding: 0 0 25px;
    margin-bottom: 30px;
  }

  h3 {
    font-size: 21px;
    font-weight: 400;
    color: ${palette('grey', 9)};
    margin: 0;
  }
`;

const PageSubTitle = styled.p`
  font-size: 13px;
  font-weight: 400;
  color: ${palette('grey', 5)};
  line-height: 1.5;
  margin-top: 5px;
  margin-bottom: 0;
`;

export { PageTitle, PageSubTitle };
